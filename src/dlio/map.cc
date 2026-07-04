/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 * Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   *
 * Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         *
 *                                                         *
 * SLC loop-closure integration: Hermes + binder           *
 *   Ported from SimpleLoopClosure (kamibukuro5656)        *
 *   ISAM2 + ICP + KD-tree loop closure                    *
 *                                                         *
 ***********************************************************/

#include "dlio/map.h"
#include "dlio/utils.h"

// ── Helper ──────────────────────────────────────────

static void makeDirectory(const std::string& directory) {
  if (!std::filesystem::is_directory(directory) || !std::filesystem::exists(directory)) {
    std::filesystem::create_directory(directory);
  }
}

// ── Constructor ─────────────────────────────────────

dlio::MapNode::MapNode() : Node("dlio_map_node") {

  this->getParams();

  // Publishers
  pub_map_       = this->create_publisher<sensor_msgs::msg::PointCloud2>("/dlio_map", 100);
  pub_vis_graph_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/vis_pose_graph", 10);
  pub_pgo_odom_  = this->create_publisher<nav_msgs::msg::Odometry>("/pgo_odom", 10);

  // Synchronized subscribers — use DLIO topics directly (no remap needed)
  sub_cloud_.subscribe(this, "/dlio/odom_node/pointcloud/deskewed");
  sub_odom_.subscribe(this,  "/dlio/odom_node/odom");

  sync_ = std::make_shared<Sync>(SyncPolicy(50), sub_cloud_, sub_odom_);
  sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(time_stamp_tolerance_));
  sync_->registerCallback(std::bind(&MapNode::cloudAndOdomCallback, this,
                                    std::placeholders::_1, std::placeholders::_2));

  // Save request topic (SLC save)
  sub_save_req_ = this->create_subscription<std_msgs::msg::String>(
      "/save_req", 10, std::bind(&MapNode::saveCallback, this, std::placeholders::_1));

  // DLIO save_pcd service
  auto save_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  save_pcd_srv_ = this->create_service<direct_lidar_inertial_odometry::srv::SavePCD>(
      "save_pcd",
      std::bind(&MapNode::savePCD, this, std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default, save_cb_group);

  // ICP setup
  icp_.setMaximumIterations(50);
  icp_.setMaxCorrespondenceDistance(search_radius_ * 2.0);
  icp_.setTransformationEpsilon(0.0001);
  icp_.setEuclideanFitnessEpsilon(0.0001);
  icp_.setRANSACIterations(0);

  // Voxel filters
  vg_target_.setLeafSize(target_voxel_leaf_size_,  target_voxel_leaf_size_,  target_voxel_leaf_size_);
  vg_source_.setLeafSize(source_voxel_leaf_size_,  source_voxel_leaf_size_,  source_voxel_leaf_size_);
  vg_map_.setLeafSize(    vis_map_voxel_leaf_size_, vis_map_voxel_leaf_size_, vis_map_voxel_leaf_size_);

  // State
  added_odom_id_    = 0;
  searched_loop_id_ = 0;
  saving_           = false;

  // ISAM2
  gtsam::ISAM2Params params;
  params.relinearizeThreshold = 0.01;
  params.relinearizeSkip      = 1;
  isam2_ = gtsam::ISAM2(params);

  Eigen::VectorXd prior_noise_vec(6);
  prior_noise_vec << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
  prior_noise_ = gtsam::noiseModel::Diagonal::Variances(prior_noise_vec);

  Eigen::VectorXd odom_noise_vec(6);
  odom_noise_vec << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
  odom_noise_ = gtsam::noiseModel::Diagonal::Variances(odom_noise_vec);

  // Colors
  odom_edge_color_.r = 0.0;  odom_edge_color_.g = 0.75; odom_edge_color_.b = 1.0;  odom_edge_color_.a = 1.0;
  loop_edge_color_.r = 1.0;  loop_edge_color_.g = 0.75; loop_edge_color_.b = 0.0;  loop_edge_color_.a = 1.0;
  node_color_.r      = 0.5;  node_color_.g      = 1.0;  node_color_.b      = 0.0;  node_color_.a      = 1.0;
  edge_scale_ = 0.1;
  node_scale_ = 0.15;

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

  // Threads
  lc_thread_  = std::thread(&MapNode::loopCloseThread, this);
  viz_thread_ = std::thread(&MapNode::visualizeThread, this);

  RCLCPP_INFO(this->get_logger(), "DLIO Map Node with Loop Closure Initialized.");
}

dlio::MapNode::~MapNode() {
  stop_lc_thread_  = true;
  stop_viz_thread_ = true;
  if (lc_thread_.joinable())  lc_thread_.join();
  if (viz_thread_.joinable()) viz_thread_.join();
  if (save_thread_.joinable()) save_thread_.join();
}

// ── Parameters ──────────────────────────────────────

void dlio::MapNode::getParams() {

  this->declare_parameter<bool>("mapped_cloud", true);
  this->declare_parameter<double>("time_stamp_tolerance", 0.01);
  this->declare_parameter<double>("keyframe_dist_th", 0.5);
  this->declare_parameter<double>("keyframe_angular_dist_th", 0.3);
  this->declare_parameter<double>("loop_search_time_diff_th", 30.0);
  this->declare_parameter<double>("loop_search_dist_diff_th", 30.0);
  this->declare_parameter<double>("loop_search_angular_dist_th", 3.14);
  this->declare_parameter<int>("loop_search_frame_interval", 1);
  this->declare_parameter<double>("search_radius", 15.0);
  this->declare_parameter<int>("target_frame_num", 50);
  this->declare_parameter<double>("target_voxel_leaf_size", 0.4);
  this->declare_parameter<double>("source_voxel_leaf_size", 0.4);
  this->declare_parameter<double>("vis_map_voxel_leaf_size", 0.8);
  this->declare_parameter<double>("fitness_score_th", 0.3);
  this->declare_parameter<int>("vis_map_cloud_frame_interval", 3);
  this->declare_parameter<double>("map/sparse/leafSize", 0.5);

  this->get_parameter("mapped_cloud", mapped_cloud_);
  this->get_parameter("time_stamp_tolerance", time_stamp_tolerance_);
  this->get_parameter("keyframe_dist_th", keyframe_dist_th_);
  this->get_parameter("keyframe_angular_dist_th", keyframe_angular_dist_th_);
  this->get_parameter("loop_search_time_diff_th", loop_search_time_diff_th_);
  this->get_parameter("loop_search_dist_diff_th", loop_search_dist_diff_th_);
  this->get_parameter("loop_search_angular_dist_th", loop_search_angular_dist_th_);
  this->get_parameter("loop_search_frame_interval", loop_search_frame_interval_);
  this->get_parameter("search_radius", search_radius_);
  this->get_parameter("target_frame_num", target_frame_num_);
  this->get_parameter("target_voxel_leaf_size", target_voxel_leaf_size_);
  this->get_parameter("source_voxel_leaf_size", source_voxel_leaf_size_);
  this->get_parameter("vis_map_voxel_leaf_size", vis_map_voxel_leaf_size_);
  this->get_parameter("fitness_score_th", fitness_score_th_);
  this->get_parameter("vis_map_cloud_frame_interval", vis_map_cloud_frame_interval_);
  this->get_parameter("map/sparse/leafSize", leaf_size_);

  search_radius_ *= search_radius_;  // store as squared
}

// ── Synchronized callback ───────────────────────────

void dlio::MapNode::cloudAndOdomCallback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg,
    const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg)
{
  PointCloudType::Ptr cloud_curr(new PointCloudType);
  pcl::fromROSMsg(*cloud_msg, *cloud_curr);

  Affine3dPtr affine_curr(new Eigen::Affine3d);
  // geometry_msgs::Pose → Eigen::Affine3d (hand-written, no tf2)
  {
    const auto& p = odom_msg->pose.pose.position;
    const auto& o = odom_msg->pose.pose.orientation;
    Eigen::Translation3d t(p.x, p.y, p.z);
    Eigen::Quaterniond q(o.w, o.x, o.y, o.z);
    *affine_curr = t * q;
  }

  publishPgoOdom(*affine_curr, *odom_msg);

  // Keyframe selection
  if (!keyframes_odom_.empty()) {
    MtxLockGuard guard(mtx_buf_);
    double dist  = (keyframes_odom_.back()->translation() - affine_curr->translation()).norm();
    Eigen::Quaterniond q_prev(keyframes_odom_.back()->rotation());
    Eigen::Quaterniond q_curr(affine_curr->rotation());
    double angle  = q_prev.angularDistance(q_curr);

    if (dist < keyframe_dist_th_ && angle < keyframe_angular_dist_th_)
      return;
  } else {
    odom_frame_id_ = odom_msg->header.frame_id;
  }

  // Map cloud to base frame
  if (mapped_cloud_) {
    PointCloudType::Ptr cloud_base(new PointCloudType);
    Eigen::Affine3d affine_inv = affine_curr->inverse();
    pcl::transformPointCloud(*cloud_curr, *cloud_base, affine_inv);
    cloud_curr = cloud_base;
  }

  // Push keyframe
  {
    MtxLockGuard guard(mtx_buf_);
    if (trajectory_dist_.empty())
      trajectory_dist_.push_back(0.0);
    else
      trajectory_dist_.push_back(trajectory_dist_.back() +
          (keyframes_odom_.back()->translation() - affine_curr->translation()).norm());

    keyframes_cloud_.push_back(cloud_curr);
    keyframes_odom_.push_back(affine_curr);
  }
}

// ── Publish PGO odometry ────────────────────────────

void dlio::MapNode::publishPgoOdom(const Eigen::Affine3d& affine_curr,
                                   const nav_msgs::msg::Odometry& odom_msg)
{
  Eigen::Affine3d pgo_affine;
  Eigen::Affine3d optimized_pose_last;
  int optimized_pose_id_last;
  getLastPose(optimized_pose_last, optimized_pose_id_last);

  if (optimized_pose_id_last != 0) {
    MtxLockGuard guard(mtx_buf_);
    pgo_affine = optimized_pose_last *
        (keyframes_odom_[optimized_pose_id_last]->inverse() * affine_curr);
  } else {
    pgo_affine = affine_curr;
  }

  nav_msgs::msg::Odometry pgo_odom_msg = odom_msg;
  // Eigen::Affine3d → geometry_msgs::Pose (hand-written, no tf2)
  Eigen::Quaterniond q(pgo_affine.rotation());
  pgo_odom_msg.pose.pose.position.x    = pgo_affine.translation().x();
  pgo_odom_msg.pose.pose.position.y    = pgo_affine.translation().y();
  pgo_odom_msg.pose.pose.position.z    = pgo_affine.translation().z();
  pgo_odom_msg.pose.pose.orientation.x = q.x();
  pgo_odom_msg.pose.pose.orientation.y = q.y();
  pgo_odom_msg.pose.pose.orientation.z = q.z();
  pgo_odom_msg.pose.pose.orientation.w = q.w();

  pub_pgo_odom_->publish(pgo_odom_msg);
}

// ── Map building & publishing ───────────────────────

dlio::MapNode::PointCloudType::Ptr dlio::MapNode::buildMapCloud(const int interval) {
  int opt_size = 0;
  {
    MtxLockGuard guard(mtx_res_);
    opt_size = opt_result_.size();
  }
  if (opt_size <= 0) return nullptr;

  PointCloudType::Ptr map(new PointCloudType);
  for (int i = 0; i < opt_size; i += (interval + 1)) {
    Eigen::Affine3d pose;
    {
      MtxLockGuard guard(mtx_res_);
      pose = Eigen::Affine3d(opt_result_.at<gtsam::Pose3>(i).matrix());
    }
    PointCloudType transformed;
    {
      MtxLockGuard guard(mtx_buf_);
      pcl::transformPointCloud(*keyframes_cloud_[i], transformed, pose);
    }
    *map += transformed;
  }
  return map;
}

void dlio::MapNode::publishMapCloud(const PointCloudType::Ptr& map) {
  if (!map || map->empty()) return;

  PointCloudType map_ds;
  vg_map_.setInputCloud(map);
  vg_map_.filter(map_ds);

  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(map_ds, msg);
  msg.header.stamp    = this->now();
  msg.header.frame_id = odom_frame_id_;

  pub_map_->publish(msg);
}

// ── Keyframe copy ───────────────────────────────────

void dlio::MapNode::copyKeyframes() {
  MtxLockGuard guard(mtx_buf_);
  for (size_t i = keyframes_cloud_copied_.size(); i < keyframes_cloud_.size(); ++i) {
    keyframes_cloud_copied_.push_back(keyframes_cloud_[i]);
    keyframes_odom_copied_.push_back(keyframes_odom_[i]);
    trajectory_dist_copied_.push_back(trajectory_dist_[i]);
  }
}

// ── Last optimized pose ─────────────────────────────

void dlio::MapNode::getLastPose(Eigen::Affine3d& pose, int& id) {
  MtxLockGuard guard(mtx_res_);
  if (!opt_result_.empty()) {
    pose = opt_result_.at<gtsam::Pose3>(opt_result_.size() - 1).matrix();
    id   = opt_result_.size() - 1;
  } else {
    MtxLockGuard guard2(mtx_buf_);
    if (!keyframes_odom_.empty()) {
      pose = *keyframes_odom_[0];
      id   = 0;
    } else {
      pose = Eigen::Affine3d();
      id   = 0;
    }
  }
}

// ── Odometry factor graph ───────────────────────────

bool dlio::MapNode::buildOdomGraph(gtsam::NonlinearFactorGraph& graph,
                                   gtsam::Values& init_estimate) {
  if (added_odom_id_ >= keyframes_odom_copied_.size())
    return false;

  if (added_odom_id_ == 0) {
    gtsam::Pose3 pose(keyframes_odom_copied_[0]->matrix());
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, pose, prior_noise_));
    init_estimate.insert(0, pose);
    added_odom_id_ = 1;
  }

  Eigen::Affine3d opt_last;
  int opt_last_id;
  getLastPose(opt_last, opt_last_id);

  for (size_t i = added_odom_id_; i < keyframes_odom_copied_.size(); ++i) {
    Eigen::Affine3d diff = keyframes_odom_copied_[i-1]->inverse() * (*keyframes_odom_copied_[i]);
    graph.add(gtsam::BetweenFactor<gtsam::Pose3>(i-1, i, gtsam::Pose3(diff.matrix()), odom_noise_));

    Eigen::Affine3d init = opt_last *
        (keyframes_odom_copied_[opt_last_id]->inverse() * (*keyframes_odom_copied_[i]));
    init_estimate.insert(i, gtsam::Pose3(init.matrix()));
    added_odom_id_++;
  }
  return true;
}

// ── KD-tree ─────────────────────────────────────────

bool dlio::MapNode::buildKDTreeMat(KDTreeMatrix& mat) {
  MtxLockGuard guard(mtx_res_);
  if (opt_result_.empty()) return false;

  mat.resize(opt_result_.size(), 3);
  for (size_t i = 0; i < opt_result_.size(); ++i)
    mat.row(i) = opt_result_.at<gtsam::Pose3>(i).translation();

  return true;
}

int dlio::MapNode::searchTarget(const KDTree& kdtree, int id_query,
                                const Eigen::Affine3d& pose_query,
                                const rclcpp::Time& stamp_query) {
  std::vector<nanoflann::ResultItem<long int, double>> ret_matches;
  nanoflann::SearchParameters sp(0, true);
  (void)kdtree.index_->radiusSearch(pose_query.translation().data(), search_radius_, ret_matches, sp);

  Eigen::Quaterniond q_query(pose_query.rotation());

  for (size_t i = 0; i < ret_matches.size(); ++i) {
    int tmp_id = ret_matches[i].first;
    rclcpp::Time tmp_stamp = pcl_conversions::fromPCL(keyframes_cloud_copied_[tmp_id]->header.stamp);
    double time_diff = std::fabs(stamp_query.seconds() - tmp_stamp.seconds());

    Eigen::Affine3d affine_target;
    {
      MtxLockGuard guard(mtx_res_);
      affine_target = Eigen::Affine3d(opt_result_.at<gtsam::Pose3>(tmp_id).matrix());
    }
    Eigen::Quaterniond q_target(affine_target.rotation());
    double angle_diff = q_query.angularDistance(q_target);
    double dist_diff  = std::fabs(trajectory_dist_copied_[id_query] - trajectory_dist_copied_[tmp_id]);

    if (time_diff  > loop_search_time_diff_th_ &&
        angle_diff < loop_search_angular_dist_th_ &&
        dist_diff  > loop_search_dist_diff_th_) {
      return tmp_id;
    }
  }
  return -1;
}

// ── Target / source cloud construction ──────────────

dlio::MapNode::PointCloudType::Ptr dlio::MapNode::buildTargetCloud(int target_id) {
  PointCloudType::Ptr target(new PointCloudType);
  PointCloudType::Ptr target_ds(new PointCloudType);

  int start = std::max(0, target_id - target_frame_num_);
  int end   = std::min((int)opt_result_.size() - 1, target_id + target_frame_num_);

  for (int i = start; i <= end; ++i) {
    Eigen::Affine3d tmp_affine;
    {
      MtxLockGuard guard(mtx_res_);
      tmp_affine = Eigen::Affine3d(opt_result_.at<gtsam::Pose3>(i).matrix());
    }
    PointCloudType tmp;
    pcl::transformPointCloud(*keyframes_cloud_copied_[i], tmp, tmp_affine);
    *target += tmp;
  }

  if (target_voxel_leaf_size_ <= 0.0) {
    target_ds = target;
  } else {
    vg_target_.setInputCloud(target);
    vg_target_.filter(*target_ds);
  }
  return target_ds;
}

dlio::MapNode::PointCloudType::Ptr dlio::MapNode::buildSourceCloud(int source_id) {
  PointCloudType::Ptr src_ds(new PointCloudType);
  if (source_voxel_leaf_size_ <= 0.0) {
    *src_ds = *keyframes_cloud_copied_[source_id];
  } else {
    vg_source_.setInputCloud(keyframes_cloud_copied_[source_id]);
    vg_source_.filter(*src_ds);
  }
  return src_ds;
}

// ── ICP registration ────────────────────────────────

bool dlio::MapNode::tryRegister(const Eigen::Affine3d& init_pose,
                                const PointCloudType::Ptr& source,
                                const PointCloudType::Ptr& target,
                                Eigen::Affine3d& result, double& score) {
  PointCloudType::Ptr unused(new PointCloudType);
  icp_.setInputSource(source);
  icp_.setInputTarget(target);
  icp_.align(*unused, init_pose.matrix().cast<float>());

  result = icp_.getFinalTransformation().cast<double>();
  score  = icp_.getFitnessScore();

  return icp_.hasConverged() && score < fitness_score_th_;
}

// ── Loop edge construction ──────────────────────────

bool dlio::MapNode::buildLoopEdge(gtsam::NonlinearFactorGraph& graph) {
  if (searched_loop_id_ >= keyframes_cloud_copied_.size())
    return false;

  KDTreeMatrix kdtree_mat;
  if (!buildKDTreeMat(kdtree_mat))
    return false;

  KDTree kdtree(3, std::cref(kdtree_mat), 10);
  kdtree.index_->buildIndex();

  Eigen::Affine3d opt_last;
  int opt_last_id;
  getLastPose(opt_last, opt_last_id);

  for (size_t i = searched_loop_id_; i < keyframes_cloud_copied_.size();
       i += (loop_search_frame_interval_ + 1))
  {
    Eigen::Affine3d pose_query = opt_last *
        (keyframes_odom_copied_[opt_last_id]->inverse() * (*keyframes_odom_copied_[i]));
    rclcpp::Time stamp_query = pcl_conversions::fromPCL(keyframes_cloud_copied_[i]->header.stamp);

    int target_id = searchTarget(kdtree, i, pose_query, stamp_query);
    if (target_id < 0) continue;

    PointCloudType::Ptr target_cloud = buildTargetCloud(target_id);
    PointCloudType::Ptr source_cloud = buildSourceCloud(i);

    Eigen::Affine3d reg_result;
    double fitness;
    if (!tryRegister(pose_query, source_cloud, target_cloud, reg_result, fitness))
      continue;

    Eigen::VectorXd noise_vec(6);
    noise_vec << fitness, fitness, fitness, fitness, fitness, fitness;
    auto constraint_noise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Cauchy::Create(1),
        gtsam::noiseModel::Diagonal::Variances(noise_vec));

    Eigen::Affine3d target_pose;
    {
      MtxLockGuard guard(mtx_res_);
      target_pose = Eigen::Affine3d(opt_result_.at<gtsam::Pose3>(target_id).matrix());
    }
    Eigen::Affine3d diff = reg_result.inverse() * target_pose;
    graph.add(gtsam::BetweenFactor<gtsam::Pose3>(i, target_id, gtsam::Pose3(diff.matrix()), constraint_noise));

    RCLCPP_INFO_STREAM(this->get_logger(), "Loop Detected: " << i << " -> " << target_id);

    {
      MtxLockGuard guard(mtx_res_);
      loop_edges_.push_back(LoopEdgeID(i, target_id));
    }
  }

  searched_loop_id_ = keyframes_cloud_copied_.size();
  return true;
}

// ── ISAM2 update ────────────────────────────────────

bool dlio::MapNode::updateISAM2(const gtsam::NonlinearFactorGraph& graph,
                                const gtsam::Values& init_estimate) {
  if (graph.empty()) return false;

  if (init_estimate.empty())
    isam2_.update(graph);
  else
    isam2_.update(graph, init_estimate);
  isam2_.update();

  {
    MtxLockGuard guard(mtx_res_);
    opt_result_ = isam2_.calculateEstimate();
  }
  return true;
}

// ── Visualization thread (1 Hz) ─────────────────────

void dlio::MapNode::visualizeThread() {
  rclcpp::Rate rate(1);
  while (rclcpp::ok() && !stop_viz_thread_) {
    PointCloudType::Ptr map = buildMapCloud(vis_map_cloud_frame_interval_);
    publishMapCloud(map);
    publishVisGraph();
    rate.sleep();
  }
}

// ── Loop-closure thread (1 Hz) ──────────────────────

void dlio::MapNode::loopCloseThread() {
  rclcpp::Rate rate(1);
  while (rclcpp::ok() && !stop_lc_thread_) {
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values init_estimate;

    copyKeyframes();
    buildOdomGraph(graph, init_estimate);
    buildLoopEdge(graph);
    updateISAM2(graph, init_estimate);

    rate.sleep();
  }
}

// ── Visualization helpers ───────────────────────────

void dlio::MapNode::buildVisOdomEdges(int n, const std_msgs::msg::Header& h,
                                      visualization_msgs::msg::Marker& m) {
  m.header        = h;
  m.ns            = "odom_edges";
  m.action        = visualization_msgs::msg::Marker::ADD;
  m.pose.orientation.w = 1.0;
  m.id            = 0;
  m.type          = visualization_msgs::msg::Marker::LINE_LIST;
  m.scale.x       = edge_scale_;
  m.color         = odom_edge_color_;
  m.points.reserve(n * 2);

  for (int i = 0; i < n - 1; ++i) {
    MtxLockGuard guard(mtx_res_);
    gtsam::Pose3 p1 = opt_result_.at<gtsam::Pose3>(i);
    gtsam::Pose3 p2 = opt_result_.at<gtsam::Pose3>(i + 1);
    geometry_msgs::msg::Point pt;
    pt.x = p1.x(); pt.y = p1.y(); pt.z = p1.z(); m.points.push_back(pt);
    pt.x = p2.x(); pt.y = p2.y(); pt.z = p2.z(); m.points.push_back(pt);
  }
}

void dlio::MapNode::buildVisLoopEdges(int n, const std_msgs::msg::Header& h,
                                      visualization_msgs::msg::Marker& m) {
  int edge_num;
  {
    MtxLockGuard guard(mtx_res_);
    edge_num = loop_edges_.size();
  }
  m.header        = h;
  m.ns            = "loop_edges";
  m.action        = visualization_msgs::msg::Marker::ADD;
  m.pose.orientation.w = 1.0;
  m.id            = 0;
  m.type          = visualization_msgs::msg::Marker::LINE_LIST;
  m.scale.x       = edge_scale_;
  m.color         = loop_edge_color_;
  m.points.reserve(edge_num * 2);

  for (int i = 0; i < edge_num; ++i) {
    MtxLockGuard guard(mtx_res_);
    int id1 = loop_edges_[i].first;
    int id2 = loop_edges_[i].second;
    if (id1 >= n || id2 >= n) continue;
    gtsam::Pose3 p1 = opt_result_.at<gtsam::Pose3>(id1);
    gtsam::Pose3 p2 = opt_result_.at<gtsam::Pose3>(id2);
    geometry_msgs::msg::Point pt;
    pt.x = p1.x(); pt.y = p1.y(); pt.z = p1.z(); m.points.push_back(pt);
    pt.x = p2.x(); pt.y = p2.y(); pt.z = p2.z(); m.points.push_back(pt);
  }
}

void dlio::MapNode::buildVisNodes(int n, const std_msgs::msg::Header& h,
                                  visualization_msgs::msg::Marker& m) {
  m.header        = h;
  m.ns            = "nodes";
  m.action        = visualization_msgs::msg::Marker::ADD;
  m.pose.orientation.w = 1.0;
  m.id            = 0;
  m.type          = visualization_msgs::msg::Marker::SPHERE_LIST;
  m.scale.x       = node_scale_;
  m.scale.y       = node_scale_;
  m.scale.z       = node_scale_;
  m.color         = node_color_;
  m.points.reserve(n);

  for (int i = 0; i < n; ++i) {
    MtxLockGuard guard(mtx_res_);
    gtsam::Pose3 p = opt_result_.at<gtsam::Pose3>(i);
    geometry_msgs::msg::Point pt;
    pt.x = p.x(); pt.y = p.y(); pt.z = p.z();
    m.points.push_back(pt);
  }
}

void dlio::MapNode::publishVisGraph() {
  int opt_size;
  {
    MtxLockGuard guard(mtx_res_);
    opt_size = opt_result_.size();
  }
  if (opt_size <= 0) return;

  std_msgs::msg::Header header;
  header.stamp    = this->now();
  header.frame_id = odom_frame_id_;

  visualization_msgs::msg::MarkerArray arr;
  arr.markers.resize(3);
  buildVisOdomEdges(opt_size, header, arr.markers[0]);
  buildVisLoopEdges(opt_size, header, arr.markers[1]);
  buildVisNodes(opt_size, header, arr.markers[2]);

  pub_vis_graph_->publish(arr);
}

// ── Save callback (SLC style) ───────────────────────

void dlio::MapNode::saveCallback(const std_msgs::msg::String::ConstSharedPtr& dir) {
  if (saving_) {
    RCLCPP_WARN_STREAM(this->get_logger(), "Already saving. Request denied.");
    return;
  }
  save_directory_ = dir->data;
  if (!save_directory_.empty() && save_directory_.back() != '/')
    save_directory_ += '/';

  RCLCPP_INFO_STREAM(this->get_logger(), "Start saving to " << save_directory_);

  if (save_thread_.joinable()) save_thread_.join();
  saving_ = true;
  save_thread_ = std::thread(&MapNode::saveThread, this);
}

void dlio::MapNode::saveFrames() {
  std::string frames_dir = save_directory_ + "frames/";
  makeDirectory(frames_dir);

  int opt_size = 0;
  {
    MtxLockGuard guard(mtx_res_);
    opt_size = opt_result_.size();
  }
  if (opt_size <= 0) {
    RCLCPP_WARN(this->get_logger(), "No optimization results to save.");
    return;
  }

  int digits = std::max(6, (int)std::to_string(opt_size).length());

  std::string csv_path = save_directory_ + "poses.csv";
  std::ofstream csv(csv_path);
  csv << "index, timestamp, x, y, z, qx, qy, qz, qw\n";

  for (int i = 0; i < opt_size; ++i) {
    Eigen::Affine3d pose;
    {
      MtxLockGuard guard(mtx_res_);
      pose = Eigen::Affine3d(opt_result_.at<gtsam::Pose3>(i).matrix());
    }
    PointCloudType copied;
    {
      MtxLockGuard guard(mtx_buf_);
      copied = *keyframes_cloud_[i];
    }

    std::stringstream fname;
    fname << std::setfill('0') << std::setw(digits) << i << ".pcd";
    Eigen::Quaterniond quat(pose.rotation());
    pcl::io::savePCDFileBinary(frames_dir + fname.str(), copied);

    csv << i << ", "
        << pcl_conversions::fromPCL(copied.header.stamp).seconds() << ", "
        << pose.translation().x() << ", " << pose.translation().y() << ", " << pose.translation().z() << ", "
        << quat.x() << ", " << quat.y() << ", " << quat.z() << ", " << quat.w() << "\n";
  }
  csv.close();
}

void dlio::MapNode::saveThread() {
  PointCloudType::Ptr map = buildMapCloud();
  if (!map || map->empty()) {
    RCLCPP_WARN(this->get_logger(), "Map is empty, nothing to save.");
    saving_ = false;
    return;
  }

  saveFrames();

  std::cout << "SAVE DIRECTORY: " << save_directory_ << std::endl;
  pcl::io::savePCDFileBinary(save_directory_ + "map.pcd", *map);
  RCLCPP_INFO(this->get_logger(), "Save completed.");
  saving_ = false;
}

// ── DLIO save_pcd service ───────────────────────────

void dlio::MapNode::savePCD(
    std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Request> req,
    std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Response> res)
{
  PointCloudType::Ptr map = buildMapCloud();
  if (!map || map->empty()) {
    res->success = false;
    RCLCPP_WARN(this->get_logger(), "save_pcd: map is empty");
    return;
  }

  float lsize = req->leaf_size;
  std::string path = req->save_path;

  std::cout << std::setprecision(2) << "Saving map to " << path + "/dlio_map.pcd"
            << " with leaf size " << to_string_with_precision(lsize, 2) << "... ";
  std::cout.flush();

  pcl::VoxelGrid<PointType> vg;
  vg.setLeafSize(lsize, lsize, lsize);
  vg.setInputCloud(map);
  vg.filter(*map);

  int ret = pcl::io::savePCDFileBinary(path + "/dlio_map.pcd", *map);
  res->success = (ret == 0);

  if (res->success) {
    std::cout << "done" << std::endl;
  } else {
    std::cout << "failed" << std::endl;
  }
}
