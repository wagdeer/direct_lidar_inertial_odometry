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
 ***********************************************************/

#include "dlio/odom.h"
#include "dlio/utils.h"

#include <algorithm>
#include <queue>
#include <omp.h>

#include "rclcpp/qos.hpp"

dlio::OdomNode::OdomNode() : Node("dlio_odom_node") {

  this->getParams();

  // Allocate threads between thread pool and OpenMP to avoid oversubscription.
  // Pool handles parallel-for (deskew) and async tasks (submap, publish, metrics).
  // OpenMP is used internally by PCL/Eigen for compute-bound operations.
  unsigned int hardware_threads = std::thread::hardware_concurrency();
  if (hardware_threads == 0) { hardware_threads = 4; } // fallback

  // Pool: at least 2, at most half of cores (but leave room for OMP)
  unsigned int pool_threads = std::max(2u, hardware_threads / 2);

  // OMP: use remaining cores, but at least 1
  unsigned int omp_threads = std::max(1u, (hardware_threads > pool_threads) ? (hardware_threads - pool_threads) / 2 : 1u);

  thread_pool.reset(pool_threads);

  // OpenMP: primary control is now in dlio.launch.py (env vars set before process start).
  // Runtime calls below are a secondary safety net.
  omp_set_num_threads(omp_threads);
  omp_set_dynamic(0);
  omp_set_nested(0);
  omp_set_max_active_levels(1);
  this->num_threads_ = omp_get_max_threads();
  std::cout << "hardware_threads: " << hardware_threads
            << " | pool: " << thread_pool.get_thread_count()
            << " | omp: " << omp_threads << std::endl;

  // Equivariant IMU preintegration: Lie group-based, replaces manual quaternion integration.
  // Noise values are conservative defaults; tune per IMU model if needed.
  this->pim_params_ = std::make_shared<Pim::Params>(
      Eigen::Vector3d(0., 0., -this->gravity_),  // gravity
      1e-4,   // gyro noise sigma
      1e-3,   // accel noise sigma
      0.,     // virtual velocity noise (unused)
      0.,     // virtual time scale noise (unused)
      1e-6,   // gyro bias random walk sigma
      1e-5,   // accel bias random walk sigma
      0., 0.  // virtual bias random walks (unused)
  );
  this->pim_.emplace(this->pim_params_);

  this->dlio_initialized = false;
  this->first_valid_scan = false;
  this->first_imu_received = false;
  this->first_keyframe_occupancy_map = false;
  if (this->imu_calibrate_) {this->imu_calibrated = false;}
  else {this->imu_calibrated = true;}
  this->deskew_status = false;
  this->deskew_size = 0;

  this->lidar_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto lidar_sub_opt = rclcpp::SubscriptionOptions();
  lidar_sub_opt.callback_group = this->lidar_cb_group;
  this->lidar_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("pointcloud", 1,
      std::bind(&dlio::OdomNode::callbackPointCloud, this, std::placeholders::_1), lidar_sub_opt);

  this->imu_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto imu_sub_opt = rclcpp::SubscriptionOptions();
  imu_sub_opt.callback_group = this->imu_cb_group;
  this->imu_sub = this->create_subscription<sensor_msgs::msg::Imu>("imu", rclcpp::SensorDataQoS(),
      std::bind(&dlio::OdomNode::callbackImu, this, std::placeholders::_1), imu_sub_opt);

  rclcpp::QoS map_qos(10);
  map_qos.transient_local();
  map_qos.reliable();
  map_qos.keep_last(1);

  this->odom_pub     = this->create_publisher<nav_msgs::msg::Odometry>("odom", 1);
  this->pose_pub     = this->create_publisher<geometry_msgs::msg::PoseStamped>("pose", 1);
  this->path_pub     = this->create_publisher<nav_msgs::msg::Path>("path", 1);
  this->kf_pose_pub  = this->create_publisher<geometry_msgs::msg::PoseArray>("kf_pose", 1);
  this->kf_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("kf_cloud", 1);
  this->deskewed_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("deskewed", 1);
  this->grid_map_pub = this->create_publisher<nav_msgs::msg::OccupancyGrid>("map", map_qos);

  this->br = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

  this->publish_timer = this->create_wall_timer(std::chrono::duration<double>(0.01), 
      std::bind(&dlio::OdomNode::publishPose, this));

  this->T = Eigen::Matrix4f::Identity();
  this->T_prior = Eigen::Matrix4f::Identity();
  this->T_corr = Eigen::Matrix4f::Identity();

  this->origin = Eigen::Vector3f(0., 0., 0.);
  this->state.p = Eigen::Vector3f(0., 0., 0.);
  this->state.q = Eigen::Quaternionf(1., 0., 0., 0.);
  this->state.v.lin.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.lin.w = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.w = Eigen::Vector3f(0., 0., 0.);

  this->lidarPose.p = Eigen::Vector3f(0., 0., 0.);
  this->lidarPose.q = Eigen::Quaternionf(1., 0., 0., 0.);

  this->imu_meas.stamp = 0.;
  this->imu_meas.ang_vel[0] = 0.;
  this->imu_meas.ang_vel[1] = 0.;
  this->imu_meas.ang_vel[2] = 0.;
  this->imu_meas.lin_accel[0] = 0.;
  this->imu_meas.lin_accel[1] = 0.;
  this->imu_meas.lin_accel[2] = 0.;

  this->imu_buffer.set_capacity(this->imu_buffer_size_);
  this->first_imu_stamp = 0.;
  this->prev_imu_stamp = 0.;
  this->imu_calib_num_samples_ = 0;
  this->imu_calib_gyro_sum_ = Eigen::Vector3f(0., 0., 0.);
  this->imu_calib_accel_sum_ = Eigen::Vector3f(0., 0., 0.);
  this->imu_calib_print_once_ = true;

  this->original_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->current_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->submap_cloud = std::make_shared<const pcl::PointCloud<PointType>>();
  this->occupancy_map_cloud_ = std::make_shared<pcl::PointCloud<PointType>>();

  this->num_processed_keyframes = 0;

  this->submap_hasChanged = true;
  this->submap_kf_idx_prev.clear();

  this->first_scan_stamp = 0.;
  this->elapsed_time = 0.;
  this->length_traversed;

  this->convex_hull.setDimension(3);
  this->concave_hull.setDimension(3);
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);
  this->concave_hull.setKeepInformation(true);

  this->gicp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp.setMaximumIterations(this->gicp_max_iter_);
  this->gicp.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp.setInitialLambdaFactor(this->gicp_init_lambda_factor_);

  this->gicp_temp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp_temp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp_temp.setMaximumIterations(this->gicp_max_iter_);
  this->gicp_temp.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp_temp.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp_temp.setInitialLambdaFactor(this->gicp_init_lambda_factor_);

  pcl::Registration<PointType, PointType>::KdTreeReciprocalPtr temp;
  this->gicp.setSearchMethodSource(temp, true);
  this->gicp.setSearchMethodTarget(temp, true);
  this->gicp_temp.setSearchMethodSource(temp, true);
  this->gicp_temp.setSearchMethodTarget(temp, true);

  this->geo.first_opt_done = false;
  this->geo.prev_vel = Eigen::Vector3f(0., 0., 0.);

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

  this->crop.setNegative(true);
  this->crop.setMin(Eigen::Vector4f(-this->crop_size_, -this->crop_size_, -this->crop_size_, 1.0));
  this->crop.setMax(Eigen::Vector4f(this->crop_size_, this->crop_size_, this->crop_size_, 1.0));

  this->voxel.setLeafSize(this->vf_res_, this->vf_res_, this->vf_res_);

  this->metrics.spaciousness.emplace_back(0.);
  this->metrics.density.emplace_back(this->gicp_max_corr_dist_);

  // CPU Specs
  char CPUBrandString[0x40];
  memset(CPUBrandString, 0, sizeof(CPUBrandString));

  this->cpu_type = "";

  #ifdef HAS_CPUID
  unsigned int CPUInfo[4] = {0,0,0,0};
  __cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
  unsigned int nExIds = CPUInfo[0];
  for (unsigned int i = 0x80000000; i <= nExIds; ++i) {
    __cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
    if (i == 0x80000002)
      memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000003)
      memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000004)
      memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
  }
  this->cpu_type = CPUBrandString;
  boost::trim(this->cpu_type);
  #endif

  FILE* file;
  struct tms timeSample;
  char line[128];

  this->lastCPU = times(&timeSample);
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;

  file = fopen("/proc/cpuinfo", "r");
  this->numProcessors = 0;
  while(fgets(line, 128, file) != nullptr) {
      if (strncmp(line, "processor", 9) == 0) this->numProcessors++;
  }
  fclose(file);

}

dlio::OdomNode::~OdomNode() {}

void dlio::OdomNode::getParams() {

  // Version
  dlio::declare_param(this, "version", this->version_, "0.0.0");

  // Frames
  dlio::declare_param(this, "frames/odom", this->odom_frame, "odom");
  dlio::declare_param(this, "frames/baselink", this->baselink_frame, "base_link");
  dlio::declare_param(this, "frames/lidar", this->lidar_frame, "lidar");
  dlio::declare_param(this, "frames/imu", this->imu_frame, "imu");

  // Deskew Flag
  dlio::declare_param(this, "pointcloud/deskew", this->deskew_, true);

  // Gravity
  dlio::declare_param(this, "odom/gravity", this->gravity_, 9.80665);

  // Compute time offset between lidar and imu
  dlio::declare_param(this, "odom/computeTimeOffset", this->time_offset_, false);

  // Keyframe Threshold
  dlio::declare_param(this, "odom/keyframe/threshD", this->keyframe_thresh_dist_, 0.1);
  dlio::declare_param(this, "odom/keyframe/threshR", this->keyframe_thresh_rot_, 1.0);

  // Submap
  dlio::declare_param(this, "odom/submap/keyframe/knn", this->submap_knn_, 10);
  dlio::declare_param(this, "odom/submap/keyframe/kcv", this->submap_kcv_, 10);
  dlio::declare_param(this, "odom/submap/keyframe/kcc", this->submap_kcc_, 10);

  // Dense map resolution
  dlio::declare_param(this, "map/dense/filtered", this->densemap_filtered_, true);

  // Wait until movement to publish map
  dlio::declare_param(this, "map/waitUntilMove", this->wait_until_move_, false);

  // Crop Box Filter
  dlio::declare_param(this, "odom/preprocessing/cropBoxFilter/size", this->crop_size_, 1.0);

  // Voxel Grid Filter
  dlio::declare_param(this, "pointcloud/voxelize", this->vf_use_, true);
  dlio::declare_param(this, "odom/preprocessing/voxelFilter/res", this->vf_res_, 0.05);

  // Adaptive Parameters
  dlio::declare_param(this, "adaptive", this->adaptive_params_, true);

  // Extrinsics
  std::vector<double> t_default{0., 0., 0.};
  std::vector<double> R_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};

  // center of gravity to imu
  std::vector<double> baselink2imu_t, baselink2imu_R;
  dlio::declare_param(this, "extrinsics/baselink2imu/t", baselink2imu_t, t_default);
  dlio::declare_param(this, "extrinsics/baselink2imu/R", baselink2imu_R, R_default);
  this->extrinsics.baselink2imu.t =
    Eigen::Vector3f(baselink2imu_t[0], baselink2imu_t[1], baselink2imu_t[2]);
  this->extrinsics.baselink2imu.R =
    Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(baselink2imu_R.begin(), baselink2imu_R.end()).data(), 3, 3);
  this->extrinsics.baselink2imu_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2imu_T.block(0, 3, 3, 1) = this->extrinsics.baselink2imu.t;
  this->extrinsics.baselink2imu_T.block(0, 0, 3, 3) = this->extrinsics.baselink2imu.R;

  // center of gravity to lidar
  std::vector<double> baselink2lidar_t, baselink2lidar_R;
  dlio::declare_param(this, "extrinsics/baselink2lidar/t", baselink2lidar_t, t_default);
  dlio::declare_param(this, "extrinsics/baselink2lidar/R", baselink2lidar_R, R_default);

  this->extrinsics.baselink2lidar.t =
    Eigen::Vector3f(baselink2lidar_t[0], baselink2lidar_t[1], baselink2lidar_t[2]);
  this->extrinsics.baselink2lidar.R =
    Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(baselink2lidar_R.begin(), baselink2lidar_R.end()).data(), 3, 3);

  this->extrinsics.baselink2lidar_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2lidar_T.block(0, 3, 3, 1) = this->extrinsics.baselink2lidar.t;
  this->extrinsics.baselink2lidar_T.block(0, 0, 3, 3) = this->extrinsics.baselink2lidar.R;

  // IMU
  dlio::declare_param(this, "odom/imu/calibration/accel", this->calibrate_accel_, true);
  dlio::declare_param(this, "odom/imu/calibration/gyro", this->calibrate_gyro_, true);
  dlio::declare_param(this, "odom/imu/calibration/time", this->imu_calib_time_, 3.0);
  dlio::declare_param(this, "odom/imu/bufferSize", this->imu_buffer_size_, 2000);

  std::vector<double> accel_default{0., 0., 0.}; std::vector<double> prior_accel_bias;
  std::vector<double> gyro_default{0., 0., 0.}; std::vector<double> prior_gyro_bias;

  dlio::declare_param(this, "odom/imu/approximateGravity", this->gravity_align_, true);
  dlio::declare_param(this, "imu/calibration", this->imu_calibrate_, true);
  dlio::declare_param(this, "imu/intrinsics/accel/bias", prior_accel_bias, accel_default);
  dlio::declare_param(this, "imu/intrinsics/gyro/bias", prior_gyro_bias, gyro_default);

  // scale-misalignment matrix
  std::vector<double> imu_sm_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};
  std::vector<double> imu_sm;

  dlio::declare_param(this, "imu/intrinsics/accel/sm", imu_sm, imu_sm_default);

  if (!this->imu_calibrate_) {
    this->state.b.accel[0] = prior_accel_bias[0];
    this->state.b.accel[1] = prior_accel_bias[1];
    this->state.b.accel[2] = prior_accel_bias[2];
    this->state.b.gyro[0] = prior_gyro_bias[0];
    this->state.b.gyro[1] = prior_gyro_bias[1];
    this->state.b.gyro[2] = prior_gyro_bias[2];
    this->imu_accel_sm_ = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(imu_sm.begin(), imu_sm.end()).data(), 3, 3);
  } else {
    this->state.b.accel = Eigen::Vector3f(0., 0., 0.);
    this->state.b.gyro = Eigen::Vector3f(0., 0., 0.);
    this->imu_accel_sm_ = Eigen::Matrix3f::Identity();
  }

  // GICP
  dlio::declare_param(this, "odom/gicp/minNumPoints", this->gicp_min_num_points_, 100);
  dlio::declare_param(this, "odom/gicp/kCorrespondences", this->gicp_k_correspondences_, 20);
  dlio::declare_param(this, "odom/gicp/maxCorrespondenceDistance", this->gicp_max_corr_dist_,
      std::sqrt(std::numeric_limits<double>::max()));
  dlio::declare_param(this, "odom/gicp/maxIterations", this->gicp_max_iter_, 64);
  dlio::declare_param(this, "odom/gicp/transformationEpsilon", this->gicp_transformation_ep_, 0.0005);
  dlio::declare_param(this, "odom/gicp/rotationEpsilon", this->gicp_rotation_ep_, 0.0005);
  dlio::declare_param(this, "odom/gicp/initLambdaFactor", this->gicp_init_lambda_factor_, 1e-9);

  // Geometric Observer
  dlio::declare_param(this, "odom/geo/Kp", this->geo_Kp_, 1.0);
  dlio::declare_param(this, "odom/geo/Kv", this->geo_Kv_, 1.0);
  dlio::declare_param(this, "odom/geo/Kq", this->geo_Kq_, 1.0);
  dlio::declare_param(this, "odom/geo/Kab", this->geo_Kab_, 1.0);
  dlio::declare_param(this, "odom/geo/Kgb", this->geo_Kgb_, 1.0);
  dlio::declare_param(this, "odom/geo/abias_max", this->geo_abias_max_, 1.0);
  dlio::declare_param(this, "odom/geo/gbias_max", this->geo_gbias_max_, 1.0);

  // Occupancy Map
  dlio::declare_param(this, "odom/occupancy/enable", this->occupancy_enable, false);
  dlio::declare_param(this, "odom/occupancy/use_dynamic_map", this->occupancy_use_dynamic_map, false);
  dlio::declare_param(this, "odom/occupancy/map_resolution", this->occupancy_map_resolution, 0.1);
  dlio::declare_param(this, "odom/occupancy/min_z", this->occupancy_min_z, 0.0);
  dlio::declare_param(this, "odom/occupancy/max_z", this->occupancy_max_z, 0.0);
}

void dlio::OdomNode::start() {

  printf("\033[2J\033[1;1H");
  std::cout << std::endl
            << "+-------------------------------------------------------------------+" << std::endl;
  std::cout << "|               Direct LiDAR-Inertial Odometry v" << this->version_  << "               |"
            << std::endl;
  std::cout << "+-------------------------------------------------------------------+" << std::endl;

}

void dlio::OdomNode::publishPose() {
  Eigen::Quaternionf q_state(this->state.q);
  Eigen::Vector3f p_state(this->state.p);
  Eigen::Vector3f v_lin(this->state.v.lin.w);
  Eigen::Vector3f v_ang(this->state.v.ang.b);
  // nav_msgs::msg::Odometry
  this->odom_ros.header.stamp = this->imu_stamp;
  this->odom_ros.header.frame_id = this->odom_frame;
  this->odom_ros.child_frame_id = this->baselink_frame;

  this->odom_ros.pose.pose.position.x = p_state[0];
  this->odom_ros.pose.pose.position.y = p_state[1];
  this->odom_ros.pose.pose.position.z = p_state[2];

  this->odom_ros.pose.pose.orientation.w = q_state.w();
  this->odom_ros.pose.pose.orientation.x = q_state.x();
  this->odom_ros.pose.pose.orientation.y = q_state.y();
  this->odom_ros.pose.pose.orientation.z = q_state.z();

  this->odom_ros.twist.twist.linear.x = v_lin[0];
  this->odom_ros.twist.twist.linear.y = v_lin[1];
  this->odom_ros.twist.twist.linear.z = v_lin[2];

  this->odom_ros.twist.twist.angular.x = v_ang[0];
  this->odom_ros.twist.twist.angular.y = v_ang[1];
  this->odom_ros.twist.twist.angular.z = v_ang[2];

  this->odom_pub->publish(this->odom_ros);

  // geometry_msgs::msg::PoseStamped
  this->pose_ros.header.stamp = this->imu_stamp;
  this->pose_ros.header.frame_id = this->odom_frame;

  this->pose_ros.pose.position.x = p_state[0];
  this->pose_ros.pose.position.y = p_state[1];
  this->pose_ros.pose.position.z = p_state[2];

  this->pose_ros.pose.orientation.w = q_state.w();
  this->pose_ros.pose.orientation.x = q_state.x();
  this->pose_ros.pose.orientation.y = q_state.y();
  this->pose_ros.pose.orientation.z = q_state.z();

  this->pose_pub->publish(this->pose_ros);

}

void dlio::OdomNode::publishToROS(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud) {
  this->publishCloud(published_cloud, T_cloud);

  // nav_msgs::msg::Path
  this->path_ros.header.stamp = this->imu_stamp;
  this->path_ros.header.frame_id = this->odom_frame;

  geometry_msgs::msg::PoseStamped p;
  p.header.stamp = this->imu_stamp;
  p.header.frame_id = this->odom_frame;
  p.pose.position.x = this->state.p[0];
  p.pose.position.y = this->state.p[1];
  p.pose.position.z = this->state.p[2];
  p.pose.orientation.w = this->state.q.w();
  p.pose.orientation.x = this->state.q.x();
  p.pose.orientation.y = this->state.q.y();
  p.pose.orientation.z = this->state.q.z();

  this->path_ros.poses.emplace_back(p);
  this->path_pub->publish(this->path_ros);

  // transform: odom to baselink
  geometry_msgs::msg::TransformStamped transformStamped;

  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->odom_frame;
  transformStamped.child_frame_id = this->baselink_frame;

  transformStamped.transform.translation.x = this->state.p[0];
  transformStamped.transform.translation.y = this->state.p[1];
  transformStamped.transform.translation.z = this->state.p[2];

  transformStamped.transform.rotation.w = this->state.q.w();
  transformStamped.transform.rotation.x = this->state.q.x();
  transformStamped.transform.rotation.y = this->state.q.y();
  transformStamped.transform.rotation.z = this->state.q.z();

  br->sendTransform(transformStamped);

  // transform: baselink to imu
  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->baselink_frame;
  transformStamped.child_frame_id = this->imu_frame;

  transformStamped.transform.translation.x = this->extrinsics.baselink2imu.t[0];
  transformStamped.transform.translation.y = this->extrinsics.baselink2imu.t[1];
  transformStamped.transform.translation.z = this->extrinsics.baselink2imu.t[2];

  Eigen::Quaternionf q(this->extrinsics.baselink2imu.R);
  transformStamped.transform.rotation.w = q.w();
  transformStamped.transform.rotation.x = q.x();
  transformStamped.transform.rotation.y = q.y();
  transformStamped.transform.rotation.z = q.z();

  br->sendTransform(transformStamped);

  // transform: baselink to lidar
  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->baselink_frame;
  transformStamped.child_frame_id = this->lidar_frame;

  transformStamped.transform.translation.x = this->extrinsics.baselink2lidar.t[0];
  transformStamped.transform.translation.y = this->extrinsics.baselink2lidar.t[1];
  transformStamped.transform.translation.z = this->extrinsics.baselink2lidar.t[2];

  Eigen::Quaternionf qq(this->extrinsics.baselink2lidar.R);
  transformStamped.transform.rotation.w = qq.w();
  transformStamped.transform.rotation.x = qq.x();
  transformStamped.transform.rotation.y = qq.y();
  transformStamped.transform.rotation.z = qq.z();

  br->sendTransform(transformStamped);
}

void dlio::OdomNode::publishCloud(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud) {
  if (this->wait_until_move_) {
    if (this->length_traversed < 0.1) { return; }
  }

  if (this->deskewed_pub->get_subscription_count()) {
    pcl::PointCloud<PointType>::Ptr deskewed_scan_t_ = std::make_shared<pcl::PointCloud<PointType>>();

    pcl::transformPointCloud (*published_cloud, *deskewed_scan_t_, T_cloud);

    sensor_msgs::msg::PointCloud2 deskewed_ros;
    pcl::toROSMsg(*deskewed_scan_t_, deskewed_ros);
    deskewed_ros.header.stamp = this->scan_header_stamp;
    deskewed_ros.header.frame_id = this->odom_frame;
    this->deskewed_pub->publish(deskewed_ros);
  }

  //this->publishKeyFrameOccupancyMap();
  //this->publishKeyFrameOccupancyMap(deskewed_scan_t_);
}

void dlio::OdomNode::publishKeyFrameOccupancyMap(pcl::PointCloud<PointType>::Ptr cloud)
{
  static int pub_freq = 0;
  static int keyframe_idx = 0;
  if (pub_freq++ % 500 != 0) {
    return; // publish every 10th frame
  }

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  if (this->keyframes.size() > keyframe_idx) {
    for (int i = keyframe_idx; i < this->keyframes.size(); i++) {
      *this->occupancy_map_cloud_ += *this->keyframes[i].second;
    }
    keyframe_idx = this->keyframes.size();
  }
  lock.unlock();
  
  std::vector<int> indices;
  pcl::PointCloud<PointType>::Ptr kf_clouds_combined(new pcl::PointCloud<PointType>());
  pcl::removeNaNFromPointCloud(*this->occupancy_map_cloud_, *kf_clouds_combined, indices);
  if (kf_clouds_combined->empty()) {
    RCLCPP_WARN(this->get_logger(), "No points in the cloud to publish grid map.");
    return;
  }

  // process occupancy map
  const float cell_size = 0.10f;

  nav_msgs::msg::OccupancyGrid grid_msg;
  grid_msg.header.stamp = this->now();
  grid_msg.header.frame_id = "map";
  grid_msg.info.map_load_time = this->now();
  grid_msg.info.resolution = cell_size;

  double x_min = std::numeric_limits<double>::max();
  double x_max = std::numeric_limits<double>::lowest();
  double y_min = std::numeric_limits<double>::max();
  double y_max = std::numeric_limits<double>::lowest();

  for (const auto & point : kf_clouds_combined->points) {
    x_min = std::min(x_min, static_cast<double>(point.x));
    x_max = std::max(x_max, static_cast<double>(point.x));
    y_min = std::min(y_min, static_cast<double>(point.y));
    y_max = std::max(y_max, static_cast<double>(point.y));
  }

  grid_msg.info.origin.position.x = x_min;
  grid_msg.info.origin.position.y = y_min;
  grid_msg.info.origin.position.z = 0.0;
  grid_msg.info.origin.orientation.x = 0.0;
  grid_msg.info.origin.orientation.y = 0.0;
  grid_msg.info.origin.orientation.z = 0.0;
  grid_msg.info.origin.orientation.w = 1.0;

  grid_msg.info.width = std::ceil((x_max - x_min) / cell_size);
  grid_msg.info.height = std::ceil((y_max - y_min) / cell_size);
  grid_msg.data.assign(grid_msg.info.width * grid_msg.info.height, 0);
  for (const auto & point : kf_clouds_combined->points) {
    if (point.z < 0.0 || point.z > 1.8) {
      continue; // filter out points that are too low or too high
    }
    int i = std::floor((point.x - x_min) / cell_size);
    int j = std::floor((point.y - y_min) / cell_size);

    if (i >= 0 && i < grid_msg.info.width && j >= 0 && j < grid_msg.info.height) {
      grid_msg.data[i + j * grid_msg.info.width] = 100;
    }
  }

  // save clear grid index
  std::vector<int> clear_grid_index(grid_msg.data.size(), 0);
  for (const auto & point : cloud->points) {
    if (point.z < 0.0 || point.z > 1.8) {
      continue; // filter out points that are too low or too high
    }
    int i = std::floor((point.x - x_min) / cell_size);
    int j = std::floor((point.y - y_min) / cell_size);
    if (i >= 0 && i < grid_msg.info.width && j >= 0 && j < grid_msg.info.height) {
      clear_grid_index[i + j * grid_msg.info.width] = 1;
    }
  }
  // set clear grid to 0
  for (int i = 0; i < grid_msg.data.size(); i++) {
    if (clear_grid_index[i] == 0) {
      grid_msg.data[i] = 0;
    }
  }

  this->grid_map_pub->publish(grid_msg);
  this->first_keyframe_occupancy_map = true;
}

void dlio::OdomNode::publishKeyFrameOccupancyMap()
{ 
  bool has_new_data = false;
  static int keyframe_idx = 0;
  static int pub_freq = 0;
  if (pub_freq++ % 90 != 0) {
    //return;
  }
  
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  if (this->keyframes.size() > keyframe_idx) {
    for (int i = keyframe_idx; i < this->keyframes.size(); i++) {
      *this->occupancy_map_cloud_ += *this->keyframes[i].second;
    }
    keyframe_idx = this->keyframes.size();
    has_new_data = true;
  }
  lock.unlock();

  if (!has_new_data) {
    return;
  }

  std::vector<int> indices;
  pcl::PointCloud<PointType>::Ptr clouds_combined(new pcl::PointCloud<PointType>());
  pcl::removeNaNFromPointCloud(*this->occupancy_map_cloud_, *clouds_combined, indices);
  if (clouds_combined->empty()) {
    RCLCPP_WARN(this->get_logger(), "No points in the cloud to publish grid map.");
    return;
  }

  publishOccupancyMap(clouds_combined);
  this->first_keyframe_occupancy_map = true;
}

void dlio::OdomNode::publishOccupancyMap(pcl::PointCloud<PointType>::Ptr cloud)
{
  const float cell_size = 0.05f;

  nav_msgs::msg::OccupancyGrid grid_msg;
  grid_msg.header.stamp = this->now();
  grid_msg.header.frame_id = "map";
  grid_msg.info.map_load_time = this->now();
  grid_msg.info.resolution = cell_size;

  double x_min = std::numeric_limits<double>::max();
  double x_max = std::numeric_limits<double>::lowest();
  double y_min = std::numeric_limits<double>::max();
  double y_max = std::numeric_limits<double>::lowest();

  if (cloud->points.empty()) {
    RCLCPP_WARN(get_logger(), "Point cloud is empty!");
    return;
  }

  for (const auto & point : cloud->points) {
    x_min = std::min(x_min, static_cast<double>(point.x));
    x_max = std::max(x_max, static_cast<double>(point.x));
    y_min = std::min(y_min, static_cast<double>(point.y));
    y_max = std::max(y_max, static_cast<double>(point.y));
  }

  grid_msg.info.origin.position.x = x_min;
  grid_msg.info.origin.position.y = y_min;
  grid_msg.info.origin.position.z = 0.0;
  grid_msg.info.origin.orientation.x = 0.0;
  grid_msg.info.origin.orientation.y = 0.0;
  grid_msg.info.origin.orientation.z = 0.0;
  grid_msg.info.origin.orientation.w = 1.0;

  grid_msg.info.width = std::ceil((x_max - x_min) / cell_size);
  grid_msg.info.height = std::ceil((y_max - y_min) / cell_size);
  grid_msg.data.assign(grid_msg.info.width * grid_msg.info.height, 0);

  for (const auto & point : cloud->points) {
    if (point.z < 0.0 || point.z > 1.8) {
      continue; // filter out points that are too low or too high
    }
    int i = std::floor((point.x - x_min) / cell_size);
    int j = std::floor((point.y - y_min) / cell_size);

    if (i >= 0 && i < grid_msg.info.width && j >= 0 && j < grid_msg.info.height) {
      grid_msg.data[i + j * grid_msg.info.width] = 100;
    }
  }

  this->grid_map_pub->publish(grid_msg);
}

void dlio::OdomNode::publishKeyframe(std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>, pcl::PointCloud<PointType>::ConstPtr> kf, rclcpp::Time timestamp) {

  // Push back
  geometry_msgs::msg::Pose p;
  Eigen::Quaternionf q_state(kf.first.second);
  Eigen::Vector3f p_state(kf.first.first);
  pcl::PointCloud<PointType>::Ptr transformed_keyframe = std::make_shared<pcl::PointCloud<PointType>>(*kf.second);
  p.position.x = p_state[0];
  p.position.y = p_state[1];
  p.position.z = p_state[2];
  p.orientation.w = q_state.w();
  p.orientation.x = q_state.x();
  p.orientation.y = q_state.y();
  p.orientation.z = q_state.z();
  this->kf_pose_ros.poses.emplace_back(p);

  // Publish
  this->kf_pose_ros.header.stamp = timestamp;
  this->kf_pose_ros.header.frame_id = this->odom_frame;
  this->kf_pose_pub->publish(this->kf_pose_ros);
  
  //publish keyframe scan for map
  if (this->vf_use_) {
    if (transformed_keyframe->points.size() == transformed_keyframe->width * transformed_keyframe->height) {
      sensor_msgs::msg::PointCloud2 keyframe_cloud_ros;
      pcl::toROSMsg(*transformed_keyframe, keyframe_cloud_ros);
      keyframe_cloud_ros.header.stamp = timestamp;
      keyframe_cloud_ros.header.frame_id = this->odom_frame;
      this->kf_cloud_pub->publish(keyframe_cloud_ros);
    }
  } else {
    sensor_msgs::msg::PointCloud2 keyframe_cloud_ros;
    pcl::toROSMsg(*transformed_keyframe, keyframe_cloud_ros);
    keyframe_cloud_ros.header.stamp = timestamp;
    keyframe_cloud_ros.header.frame_id = this->odom_frame;
    this->kf_cloud_pub->publish(keyframe_cloud_ros);
  }
}

void dlio::OdomNode::getScanFromROS(const sensor_msgs::msg::PointCloud2::SharedPtr& pc) {

  pcl::PointCloud<PointType>::Ptr original_scan_ = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::fromROSMsg(*pc, *original_scan_);

  // Remove NaNs
  std::vector<int> idx;
  original_scan_->is_dense = false;
  pcl::removeNaNFromPointCloud(*original_scan_, *original_scan_, idx);

  // Crop Box Filter
  this->crop.setInputCloud(original_scan_);
  this->crop.filter(*original_scan_);

  // only support Livox
  this->sensor = dlio::SensorType::LIVOX;
  bool has_timestamp_field = false;
  for (auto &field : pc->fields) {
    if (field.name == "timestamp") {
      has_timestamp_field = true;
      break;
    }
  }
  
  if (!has_timestamp_field) {
    this->sensor = dlio::SensorType::UNKNOWN;
    this->deskew_ = false;
  }

  this->scan_header_stamp = pc->header.stamp;
  this->original_scan = original_scan_;

}

void dlio::OdomNode::preprocessPoints() {

  // Deskew the original dlio-type scan
  if (this->deskew_) {

    this->deskewPointcloud();

    if (!this->first_valid_scan) {
      return;
    }

  } else {

    this->scan_stamp = rclcpp::Time(this->scan_header_stamp).seconds();

    // don't process scans until IMU data is present
    if (!this->first_valid_scan) {

      if (this->imu_buffer.empty() || this->scan_stamp <= this->imu_buffer.back().stamp) {
        return;
      }

      this->first_valid_scan = true;
      this->T_prior = this->T; // assume no motion for the first scan

    } else {

      // IMU prior for second scan onwards
      std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;
        frames = this->integrateImu(this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
                                  this->geo.prev_vel.cast<float>(), {this->scan_stamp});

      if (frames.size() > 0) {
        this->T_prior = frames.back();
      } else {
        this->T_prior = this->T;
      }

    }

    pcl::PointCloud<PointType>::Ptr deskewed_scan_ = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::transformPointCloud (*this->original_scan, *deskewed_scan_,
                              this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
  }

  // Voxel Grid Filter
  if (this->vf_use_) {
    pcl::PointCloud<PointType>::Ptr current_scan_ = std::make_shared<pcl::PointCloud<PointType>>();
    this->voxel.setInputCloud(this->deskewed_scan);
    this->voxel.filter(*current_scan_);
    this->current_scan = current_scan_;
  } else {
    this->current_scan = this->deskewed_scan;
  }

}

void dlio::OdomNode::deskewPointcloud() {

  pcl::PointCloud<PointType>::Ptr deskewed_scan_ = std::make_shared<pcl::PointCloud<PointType>>(1, this->original_scan->points.size());

  // individual point timestamps should be relative to this time
  double sweep_ref_time = rclcpp::Time(this->scan_header_stamp).seconds();

  std::function<bool(const PointType&, const PointType&)> point_time_cmp = 
    [](const PointType& p1, const PointType& p2) { return p1.timestamp < p2.timestamp; };
  
  std::function<bool(boost::range::index_value<PointType&, long>,
                     boost::range::index_value<PointType&, long>)> point_time_neq = 
    [](boost::range::index_value<PointType&, long> p1,
       boost::range::index_value<PointType&, long> p2) { return p1.value().timestamp != p2.value().timestamp; };
  
  std::function<double(boost::range::index_value<PointType&, long>)> extract_point_time = 
    [&sweep_ref_time](boost::range::index_value<PointType&, long> pt) { return pt.value().timestamp * 1e-9f; };

  // copy points into deskewed_scan_ in order of timestamp
  std::partial_sort_copy(this->original_scan->points.begin(), this->original_scan->points.end(),
                         deskewed_scan_->points.begin(), deskewed_scan_->points.end(), point_time_cmp);

  // filter unique timestamps
  auto points_unique_timestamps = deskewed_scan_->points
                                  | boost::adaptors::indexed()
                                  | boost::adaptors::adjacent_filtered(point_time_neq);

  // extract timestamps from points and put them in their own list
  std::vector<double> timestamps;
  std::vector<int> unique_time_indices;

  // compute offset between sweep reference time and first point timestamp
  double offset = 0.0;
  if (this->time_offset_) {
    offset = sweep_ref_time - extract_point_time(*points_unique_timestamps.begin());
  }

  // build list of unique timestamps and indices of first point with each timestamp
  for (auto it = points_unique_timestamps.begin(); it != points_unique_timestamps.end(); it++) {
    timestamps.emplace_back(extract_point_time(*it) + offset);
    unique_time_indices.emplace_back(it->index());
  }
  unique_time_indices.emplace_back(deskewed_scan_->points.size());

  int median_pt_index = timestamps.size() / 2;
  this->scan_stamp = timestamps[median_pt_index]; // set this->scan_stamp to the timestamp of the median point

  // don't process scans until IMU data is present
  if (!this->first_valid_scan) {
    if (this->imu_buffer.empty() || this->scan_stamp <= this->imu_buffer.back().stamp) {
      return;
    }

    this->first_valid_scan = true;
    this->T_prior = this->T; // assume no motion for the first scan
    pcl::transformPointCloud (*deskewed_scan_, *deskewed_scan_, this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = true;
    return;
  }

  // IMU prior & deskewing for second scan onwards
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;
  frames = this->integrateImu(this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
                              this->geo.prev_vel.cast<float>(), timestamps);

  // if there are no frames between the start and end of the sweep
  // that probably means that there's a sync issue
  if (frames.empty() || frames.size() != timestamps.size()) {
    RCLCPP_WARN(this->get_logger(), 
      "IMU-LiDAR sync issue: frames=%zu, timestamps=%zu. Using fallback transformation.", 
      frames.size(), timestamps.size());

    this->T_prior = this->T;
    pcl::transformPointCloud (*deskewed_scan_, *deskewed_scan_, this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
    return;
  }

  this->deskew_size = frames.size(); // if integration successful, equal to timestamps.size()

  // update prior to be the estimated pose at the median time of the scan (corresponds to this->scan_stamp)
  this->T_prior = frames[median_pt_index];

  const std::size_t segment_count = timestamps.size();
  const std::size_t point_count = deskewed_scan_->points.size();
  const std::size_t pool_threads = std::max<std::size_t>(1, thread_pool.get_thread_count());
  constexpr std::size_t kMinPointsPerBlock = 4096;
  const std::size_t blocks_by_points = std::max<std::size_t>(1, (point_count + kMinPointsPerBlock - 1) / kMinPointsPerBlock);
  const std::size_t num_blocks = std::min(segment_count, std::min(pool_threads, blocks_by_points));

  auto transform_segment = [this, &deskewed_scan_, &frames, &unique_time_indices](const std::size_t i) {
    if (i >= frames.size() || i >= unique_time_indices.size() - 1) {
      RCLCPP_FATAL(this->get_logger(),"Index out of bounds in deskew loop");
      return;
    }

    const Eigen::Matrix4f T = frames[i] * this->extrinsics.baselink2lidar_T;
    const int start_idx = unique_time_indices[i];
    const int end_idx = unique_time_indices[i + 1];
    if (start_idx < 0 || end_idx > static_cast<int>(deskewed_scan_->points.size()) || start_idx >= end_idx) {
      RCLCPP_FATAL(this->get_logger(),"Index out of bounds in deskewPointcloud");
      return;
    }

    // Transform points in [start_idx, end_idx) to world frame.
    for (int k = start_idx; k < end_idx; ++k) {
      auto &pt = deskewed_scan_->points[k];
      pt.getVector4fMap()[3] = 1.f;
      pt.getVector4fMap() = T * pt.getVector4fMap();
    }
  };

  // For small workloads, avoid queueing/synchronization overhead.
  if (num_blocks <= 1 || point_count < kMinPointsPerBlock) {
    for (std::size_t i = 0; i < segment_count; ++i) {
      transform_segment(i);
    }
  } else {
    const BS::multi_future<void> loop_future = thread_pool.submit_loop(0, segment_count, transform_segment, num_blocks);
    loop_future.wait();
  }

  this->deskewed_scan = deskewed_scan_;
  this->deskew_status = true;

}

void dlio::OdomNode::initializeInputTarget() {

  this->prev_scan_stamp = this->scan_stamp;

  // keep history of keyframes
  this->keyframes.emplace_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan));
  this->keyframe_timestamps.emplace_back(this->scan_header_stamp);
  this->keyframe_normals.emplace_back(this->gicp.getSourceCovariances());
  this->keyframe_transformations.emplace_back(this->T_corr);

}

void dlio::OdomNode::setInputSource() {
  this->gicp.setInputSource(this->current_scan);
  this->gicp.calculateSourceCovariances();
}

void dlio::OdomNode::initializeDLIO() {

  // Wait for IMU
  if (!this->first_imu_received || !this->imu_calibrated) {
    return;
  }

  this->dlio_initialized = true;
  std::cout << std::endl << " DLIO initialized!" << std::endl;

}

void dlio::OdomNode::callbackPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr pc) {

  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(main_loop_running_mutex);
  this->main_loop_running = true;
  lock.unlock();

  double then = this->now().seconds();

  if (this->first_scan_stamp == 0.) {
    this->first_scan_stamp = rclcpp::Time(pc->header.stamp).seconds();
  }

  // DLIO Initialization procedures (IMU calib, gravity align)
  if (!this->dlio_initialized) {
    this->initializeDLIO();
  }

  // Convert incoming scan into DLIO format
  this->getScanFromROS(pc);

  // Preprocess points
  this->preprocessPoints();

  if (!this->first_valid_scan) {
    return;
  }

  if (this->current_scan->points.size() <= this->gicp_min_num_points_) {
    RCLCPP_FATAL(this->get_logger(), "Low number of points in the cloud!");
    return;
  }

  // Compute metrics on per-frame snapshots to avoid races with async execution.
  const auto scan_snapshot = this->original_scan;
  const float density_snapshot = this->geo.first_opt_done ? this->gicp.source_density_ : 0.f;
  if (!this->metrics_task_running_.exchange(true)) {
    thread_pool.detach_task([this, scan_snapshot, density_snapshot]() {
      try {
        this->computeMetrics(scan_snapshot, density_snapshot);
      } catch (...) {
        // Keep async metrics failures from permanently disabling scheduling.
      }
      this->metrics_task_running_.store(false);
    });
  }

  // Set Adaptive Parameters
  if (this->adaptive_params_) {
    this->setAdaptiveParams();
  }

  // Set new frame as input source
  this->setInputSource();

  // Set initial frame as first keyframe
  if (this->keyframes.size() == 0) {
    this->initializeInputTarget();
    this->main_loop_running = false;
    this->submap_future = thread_pool.submit_task([this]() {
      this->buildKeyframesAndSubmap(this->state);
    });
    this->submap_future.wait(); // wait until completion
    return;
  }

  // Get the next pose via IMU + S2M + GEO
  this->getNextPose();

  // Update current keyframe poses and map
  this->updateKeyframes();

  // Build keyframe normals and submap if needed (and if we're not already waiting)
  if (this->new_submap_is_ready) {
    this->main_loop_running = false;
    this->submap_future = thread_pool.submit_task([this]() {
      this->buildKeyframesAndSubmap(this->state);
    });
  } else {
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_build_cv.notify_one();
  }

  // Update trajectory
  this->trajectory.emplace_back( std::make_pair(this->state.p, this->state.q) );

  // Update time stamps
  this->lidar_rates.emplace_back( 1. / (this->scan_stamp - this->prev_scan_stamp) );
  this->prev_scan_stamp = this->scan_stamp;
  this->elapsed_time = this->scan_stamp - this->first_scan_stamp;

  // Publish stuff to ROS
  pcl::PointCloud<PointType>::ConstPtr published_cloud;
  if (this->densemap_filtered_) {
    published_cloud = this->current_scan;
  } else {
    published_cloud = this->deskewed_scan;
  }

  // publish to ros in thread pool
  thread_pool.detach_task([this, published_cloud]() {
    this->publishToROS(published_cloud, this->T_corr);
  });

  // Update some statistics
  this->comp_times.emplace_back(this->now().seconds() - then);
  this->gicp_hasConverged = this->gicp.hasConverged();

  // debug in thread pool
  // thread_pool.detach_task([this]() {
  //   this->debug();
  // });

  this->geo.first_opt_done = true;

}

void dlio::OdomNode::callbackImu(const sensor_msgs::msg::Imu::SharedPtr imu_raw) {

  this->first_imu_received = true;

  sensor_msgs::msg::Imu::SharedPtr imu = this->transformImu( imu_raw );
  this->imu_stamp = imu->header.stamp;
  double imu_stamp_secs = rclcpp::Time(imu->header.stamp).seconds();

  Eigen::Vector3f lin_accel;
  Eigen::Vector3f ang_vel;

  // Get IMU samples
  ang_vel[0] = imu->angular_velocity.x;
  ang_vel[1] = imu->angular_velocity.y;
  ang_vel[2] = imu->angular_velocity.z;

  lin_accel[0] = imu->linear_acceleration.x;
  lin_accel[1] = imu->linear_acceleration.y;
  lin_accel[2] = imu->linear_acceleration.z;

  if (this->first_imu_stamp == 0.) {
    this->first_imu_stamp = imu_stamp_secs;
  }

  // IMU calibration procedure - do for three seconds
  if (!this->imu_calibrated) {

    if ((imu_stamp_secs - this->first_imu_stamp) < this->imu_calib_time_) {

      this->imu_calib_num_samples_++;
      this->imu_calib_gyro_sum_ += ang_vel;
      this->imu_calib_accel_sum_ += lin_accel;

      if (this->imu_calib_print_once_) {
        std::cout << std::endl << " Calibrating IMU for " << this->imu_calib_time_ << " seconds... ";
        std::cout.flush();
        this->imu_calib_print_once_ = false;
      }

    } else {

      std::cout << "done" << std::endl << std::endl;

      const int sample_count = std::max(1, this->imu_calib_num_samples_);
      const Eigen::Vector3f gyro_avg = this->imu_calib_gyro_sum_ / static_cast<float>(sample_count);
      const Eigen::Vector3f accel_avg = this->imu_calib_accel_sum_ / static_cast<float>(sample_count);

      Eigen::Vector3f grav_vec (0., 0., this->gravity_);

      if (this->gravity_align_) {

        // Estimate gravity vector - Only approximate if biases have not been pre-calibrated
        grav_vec = (accel_avg - this->state.b.accel).normalized() * abs(this->gravity_);
        Eigen::Quaternionf grav_q = Eigen::Quaternionf::FromTwoVectors(grav_vec, Eigen::Vector3f(0., 0., this->gravity_));

        // set gravity aligned orientation
        this->state.q = grav_q;
        this->T.block(0,0,3,3) = this->state.q.toRotationMatrix();
        this->lidarPose.q = this->state.q;

        // rpy
        auto euler = grav_q.toRotationMatrix().eulerAngles(2, 1, 0);
        double yaw = euler[0] * (180.0/M_PI);
        double pitch = euler[1] * (180.0/M_PI);
        double roll = euler[2] * (180.0/M_PI);

        // use alternate representation if the yaw is smaller
        if (abs(remainder(yaw + 180.0, 360.0)) < abs(yaw)) {
          yaw   = remainder(yaw + 180.0,   360.0);
          pitch = remainder(180.0 - pitch, 360.0);
          roll  = remainder(roll + 180.0,  360.0);
        }
        std::cout << " Estimated initial attitude:" << std::endl;
        std::cout << "   Roll  [deg]: " << to_string_with_precision(roll, 4) << std::endl;
        std::cout << "   Pitch [deg]: " << to_string_with_precision(pitch, 4) << std::endl;
        std::cout << "   Yaw   [deg]: " << to_string_with_precision(yaw, 4) << std::endl;
        std::cout << std::endl;
      }

      if (this->calibrate_accel_) {

        // subtract gravity from avg accel to get bias
        this->state.b.accel = accel_avg - grav_vec;

        std::cout << " Accel biases [xyz]: " << to_string_with_precision(this->state.b.accel[0], 8) << ", "
                                             << to_string_with_precision(this->state.b.accel[1], 8) << ", "
                                             << to_string_with_precision(this->state.b.accel[2], 8) << std::endl;
      }

      if (this->calibrate_gyro_) {

        this->state.b.gyro = gyro_avg;

        std::cout << " Gyro biases  [xyz]: " << to_string_with_precision(this->state.b.gyro[0], 8) << ", "
                                             << to_string_with_precision(this->state.b.gyro[1], 8) << ", "
                                             << to_string_with_precision(this->state.b.gyro[2], 8) << std::endl;
      }

      this->imu_calibrated = true;
      this->imu_calib_print_once_ = true;

    }

  } else {

    double dt = imu_stamp_secs - this->prev_imu_stamp;
    if (dt == 0) { dt = 1.0/200.0; }
    this->imu_rates.emplace_back( 1./dt );

    // Apply the calibrated bias to the new IMU measurements
    this->imu_meas.stamp = imu_stamp_secs;
    this->imu_meas.dt = dt;
    this->prev_imu_stamp = this->imu_meas.stamp;

    Eigen::Vector3f lin_accel_corrected = (this->imu_accel_sm_ * lin_accel) - this->state.b.accel;
    Eigen::Vector3f ang_vel_corrected = ang_vel - this->state.b.gyro;

    this->imu_meas.lin_accel = lin_accel_corrected;
    this->imu_meas.ang_vel = ang_vel_corrected;

    // Store calibrated IMU measurements into imu buffer for manual integration later.
    this->mtx_imu.lock();
    this->imu_buffer.push_front(this->imu_meas);
    this->mtx_imu.unlock();

    // Notify the callbackPointCloud thread that IMU data exists for this time
    this->cv_imu_stamp.notify_one();

    if (this->geo.first_opt_done) {
      // Geometric Observer: Propagate State
      this->propagateState();
    }

  }

}

void dlio::OdomNode::getNextPose() {

  // Check if the new submap is ready to be used
  this->new_submap_is_ready = (this->submap_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready);

  if (this->new_submap_is_ready && this->submap_hasChanged) {

    // Set the current global submap as the target cloud
    this->gicp.registerInputTarget(this->submap_cloud);

    // Set submap kdtree
    this->gicp.target_kdtree_ = this->submap_kdtree;

    // Set target cloud's normals as submap normals
    this->gicp.setTargetCovariances(this->submap_normals);

    this->submap_hasChanged = false;
  }

  // Align with current submap with global IMU transformation as initial guess
  pcl::PointCloud<PointType>::Ptr aligned = std::make_shared<pcl::PointCloud<PointType>>();
  this->gicp.align(*aligned);

  // Get final transformation in global frame
  this->T_corr = this->gicp.getFinalTransformation(); // "correction" transformation
  this->T = this->T_corr * this->T_prior;

  // Update next global pose
  // Both source and target clouds are in the global frame now, so tranformation is global
  this->propagateGICP();

  // Geometric observer update
  this->updateState();

}

bool dlio::OdomNode::imuMeasFromTimeRange(double start_time, double end_time,
                                          boost::circular_buffer<ImuMeas>::reverse_iterator& begin_imu_it,
                                          boost::circular_buffer<ImuMeas>::reverse_iterator& end_imu_it) {

  if (this->imu_buffer.empty() || this->imu_buffer.front().stamp < end_time) {
    // Wait for the latest IMU data
    std::unique_lock<decltype(this->mtx_imu)> lock(this->mtx_imu);
    this->cv_imu_stamp.wait(lock, [this, &end_time]{ return this->imu_buffer.front().stamp >= end_time; });
  }

  auto imu_it = this->imu_buffer.begin();

  auto last_imu_it = imu_it;
  imu_it++;
  while (imu_it != this->imu_buffer.end() && imu_it->stamp >= end_time) {
    last_imu_it = imu_it;
    imu_it++;
  }

  while (imu_it != this->imu_buffer.end() && imu_it->stamp >= start_time) {
    imu_it++;
  }

  if (imu_it == this->imu_buffer.end()) {
    // not enough IMU measurements, return false
    return false;
  }
  imu_it++;

  // Set reverse iterators (to iterate forward in time)
  end_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(last_imu_it);
  begin_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(imu_it);

  return true;
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
dlio::OdomNode::integrateImu(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init,
                             Eigen::Vector3f v_init, const std::vector<double>& sorted_timestamps) {

  const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> empty;

  if (sorted_timestamps.empty() || start_time > sorted_timestamps.front()) {
    RCLCPP_ERROR(this->get_logger(),
      "Invalid input, return empty vector: start_time=%.6f, sorted_timestamps.front()=%.6f.",
      start_time, sorted_timestamps.front());
    return empty;
  }

  boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it;
  boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it;
  if (!this->imuMeasFromTimeRange(start_time, sorted_timestamps.back(), begin_imu_it, end_imu_it)) {
    RCLCPP_WARN(this->get_logger(),
      "Insufficient IMU measurements: start=%.6f end=%.6f",
      start_time, sorted_timestamps.back());
    return empty;
  }

  // Use equivariant Lie-group preintegration instead of manual quaternion integration.
  // IMU data in imu_buffer is already bias-corrected, so use zero bias in the pim.
  pim_->resetIntegrationAndSetBias(Pim::Vec10::Zero());

  // Feed IMU measurements into the pim, capturing SE3 at each requested timestamp.
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> imu_se3;
  imu_se3.reserve(sorted_timestamps.size());

  auto stamp_it = sorted_timestamps.begin();
  auto imu_prev = begin_imu_it;
  auto imu_curr = imu_prev + 1;

  const Eigen::Matrix3f R_init = q_init.toRotationMatrix();

  // If start_time is before the first IMU sample, output identity for timestamps before it.
  while (stamp_it != sorted_timestamps.end() && *stamp_it <= imu_prev->stamp) {
    imu_se3.emplace_back(Eigen::Matrix4f::Identity());
    ++stamp_it;
  }

  for (; imu_curr != end_imu_it; ++imu_curr) {
    const ImuMeas& m = *imu_curr;
    const double dt = m.dt;

    // Feed bias-corrected IMU data: accel in m/s², gyro in rad/s
    pim_->integrateMeasurement(
        Eigen::Vector3d(m.lin_accel[0], m.lin_accel[1], m.lin_accel[2]),
        Eigen::Vector3d(m.ang_vel[0], m.ang_vel[1], m.ang_vel[2]),
        dt);

    // Output absolute world-frame SE3 for any timestamps in this IMU interval.
    while (stamp_it != sorted_timestamps.end() && *stamp_it <= m.stamp) {
      // Compose initial pose (world) with relative delta (body frame) to get absolute SE3.
      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      T.block<3,3>(0,0) = R_init * pim_->deltaRij().cast<float>();
      T.block<3,1>(0,3) = p_init + R_init * pim_->deltaPij().cast<float>();
      imu_se3.emplace_back(T);
      ++stamp_it;
    }
    imu_prev = imu_curr;
  }

  // Output identity for any remaining timestamps beyond the last IMU sample.
  while (stamp_it != sorted_timestamps.end()) {
    imu_se3.emplace_back(Eigen::Matrix4f::Identity());
    ++stamp_it;
  }

  return imu_se3;
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
dlio::OdomNode::integrateImuInternal(Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                                     const std::vector<double>& sorted_timestamps,
                                     boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it,
                                     boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it) {
  // Deprecated: kept for API compatibility, delegates to the equivariant method internally.
  // Called only from the old integrateImu (which now uses the pim_ directly).
  const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> empty;
  RCLCPP_WARN(this->get_logger(), "integrateImuInternal called — should not happen.");
  return empty;
}

void dlio::OdomNode::propagateGICP() {

  this->lidarPose.p << this->T(0,3), this->T(1,3), this->T(2,3);

  Eigen::Matrix3f rotSO3;
  rotSO3 << this->T(0,0), this->T(0,1), this->T(0,2),
            this->T(1,0), this->T(1,1), this->T(1,2),
            this->T(2,0), this->T(2,1), this->T(2,2);

  Eigen::Quaternionf q(rotSO3);

  // Normalize quaternion
  double norm = sqrt(q.w()*q.w() + q.x()*q.x() + q.y()*q.y() + q.z()*q.z());
  q.w() /= norm; q.x() /= norm; q.y() /= norm; q.z() /= norm;
  this->lidarPose.q = q;

}

void dlio::OdomNode::propagateState() {

  // Lock thread to prevent state from being accessed by UpdateState
  std::lock_guard<std::mutex> lock( this->geo.mtx );

  double dt = this->imu_meas.dt;

  Eigen::Quaternionf qhat = this->state.q, omega;
  Eigen::Vector3f world_accel;

  // Transform accel from body to world frame
  world_accel = qhat._transformVector(this->imu_meas.lin_accel);

  // Accel propogation
  this->state.p[0] += this->state.v.lin.w[0]*dt + 0.5*dt*dt*world_accel[0];
  this->state.p[1] += this->state.v.lin.w[1]*dt + 0.5*dt*dt*world_accel[1];
  this->state.p[2] += this->state.v.lin.w[2]*dt + 0.5*dt*dt*(world_accel[2] - this->gravity_);

  this->state.v.lin.w[0] += world_accel[0]*dt;
  this->state.v.lin.w[1] += world_accel[1]*dt;
  this->state.v.lin.w[2] += (world_accel[2] - this->gravity_)*dt;
  this->state.v.lin.b = this->state.q.toRotationMatrix().inverse() * this->state.v.lin.w;

  // Gyro propogation
  omega.w() = 0;
  omega.vec() = this->imu_meas.ang_vel;
  Eigen::Quaternionf tmp = qhat * omega;
  this->state.q.w() += 0.5 * dt * tmp.w();
  this->state.q.vec() += 0.5 * dt * tmp.vec();

  // Ensure quaternion is properly normalized
  this->state.q.normalize();

  this->state.v.ang.b = this->imu_meas.ang_vel;
  this->state.v.ang.w = this->state.q.toRotationMatrix() * this->state.v.ang.b;

}

void dlio::OdomNode::updateState() {

  // Lock thread to prevent state from being accessed by PropagateState
  std::lock_guard<std::mutex> lock( this->geo.mtx );

  Eigen::Vector3f pin = this->lidarPose.p;
  Eigen::Quaternionf qin = this->lidarPose.q;
  double dt = this->scan_stamp - this->prev_scan_stamp;

  Eigen::Quaternionf qe, qhat, qcorr;
  qhat = this->state.q;

  // Constuct error quaternion
  qe = qhat.conjugate()*qin;

  double sgn = 1.;
  if (qe.w() < 0) {
    sgn = -1;
  }

  // Construct quaternion correction
  qcorr.w() = 1 - abs(qe.w());
  qcorr.vec() = sgn*qe.vec();
  qcorr = qhat * qcorr;

  Eigen::Vector3f err = pin - this->state.p;
  Eigen::Vector3f err_body;

  err_body = qhat.conjugate()._transformVector(err);

  double abias_max = this->geo_abias_max_;
  double gbias_max = this->geo_gbias_max_;

  // Update accel bias
  this->state.b.accel -= dt * this->geo_Kab_ * err_body;
  this->state.b.accel = this->state.b.accel.array().min(abias_max).max(-abias_max);

  // Update gyro bias
  this->state.b.gyro[0] -= dt * this->geo_Kgb_ * qe.w() * qe.x();
  this->state.b.gyro[1] -= dt * this->geo_Kgb_ * qe.w() * qe.y();
  this->state.b.gyro[2] -= dt * this->geo_Kgb_ * qe.w() * qe.z();
  this->state.b.gyro = this->state.b.gyro.array().min(gbias_max).max(-gbias_max);

  // Update state
  this->state.p += dt * this->geo_Kp_ * err;
  this->state.v.lin.w += dt * this->geo_Kv_ * err;

  this->state.q.w() += dt * this->geo_Kq_ * qcorr.w();
  this->state.q.x() += dt * this->geo_Kq_ * qcorr.x();
  this->state.q.y() += dt * this->geo_Kq_ * qcorr.y();
  this->state.q.z() += dt * this->geo_Kq_ * qcorr.z();
  this->state.q.normalize();

  // store previous pose, orientation, and velocity
  this->geo.prev_p = this->state.p;
  this->geo.prev_q = this->state.q;
  this->geo.prev_vel = this->state.v.lin.w;

}

sensor_msgs::msg::Imu::SharedPtr dlio::OdomNode::transformImu(const sensor_msgs::msg::Imu::SharedPtr& imu_raw) {

  auto imu = std::make_shared<sensor_msgs::msg::Imu>();

  // Copy header
  imu->header = imu_raw->header;

  double imu_stamp_secs = rclcpp::Time(imu->header.stamp).seconds();
  static double prev_stamp = imu_stamp_secs;
  double dt = imu_stamp_secs - prev_stamp;
  prev_stamp = imu_stamp_secs;
  
  if (dt == 0) { dt = 1.0/200.0; }

  // Transform angular velocity (will be the same on a rigid body, so just rotate to ROS convention)
  Eigen::Vector3f ang_vel(imu_raw->angular_velocity.x,
                          imu_raw->angular_velocity.y,
                          imu_raw->angular_velocity.z);

  Eigen::Vector3f ang_vel_cg = this->extrinsics.baselink2imu.R * ang_vel;

  imu->angular_velocity.x = ang_vel_cg[0];
  imu->angular_velocity.y = ang_vel_cg[1];
  imu->angular_velocity.z = ang_vel_cg[2];

  static Eigen::Vector3f ang_vel_cg_prev = ang_vel_cg;

  // Transform linear acceleration (need to account for component due to translational difference)
  Eigen::Vector3f lin_accel(imu_raw->linear_acceleration.x,
                            imu_raw->linear_acceleration.y,
                            imu_raw->linear_acceleration.z);

  Eigen::Vector3f lin_accel_cg = this->extrinsics.baselink2imu.R * lin_accel;

  lin_accel_cg = lin_accel_cg
                 + ((ang_vel_cg - ang_vel_cg_prev) / dt).cross(-this->extrinsics.baselink2imu.t)
                 + ang_vel_cg.cross(ang_vel_cg.cross(-this->extrinsics.baselink2imu.t));

  ang_vel_cg_prev = ang_vel_cg;

  imu->linear_acceleration.x = lin_accel_cg[0];
  imu->linear_acceleration.y = lin_accel_cg[1];
  imu->linear_acceleration.z = lin_accel_cg[2];

  return imu;

}

void dlio::OdomNode::computeMetrics(const pcl::PointCloud<PointType>::ConstPtr& scan_snapshot, float density_snapshot) {
  this->computeSpaciousness(scan_snapshot);
  this->computeDensity(density_snapshot);
}

void dlio::OdomNode::computeSpaciousness(const pcl::PointCloud<PointType>::ConstPtr& scan_snapshot) {

  if (!scan_snapshot || scan_snapshot->points.empty()) {
    return;
  }

  // Compute radial distance of points in XY plane.
  std::vector<float> ds;
  ds.reserve(scan_snapshot->points.size());

  for (const auto& pt : scan_snapshot->points) {
    ds.emplace_back(std::sqrt(pt.x * pt.x + pt.y * pt.y));
  }

  const size_t mid = ds.size() / 2;
  std::nth_element(ds.begin(), ds.begin() + mid, ds.end());
  const float median_curr = ds[mid];

  std::lock_guard<std::mutex> lock(this->metrics_mutex);
  if (!this->spaciousness_lpf_prev_) {
    this->spaciousness_lpf_prev_ = median_curr;
  }
  const float median_lpf = 0.95f * this->spaciousness_lpf_prev_.value() + 0.05f * median_curr;
  this->spaciousness_lpf_prev_ = median_lpf;
  this->metrics.spaciousness.emplace_back(median_lpf);

}

void dlio::OdomNode::computeDensity(float density_curr) {
  std::lock_guard<std::mutex> lock(this->metrics_mutex);
  if (!this->density_lpf_prev_) {
    this->density_lpf_prev_ = density_curr;
  }
  const float density_lpf = 0.95f * this->density_lpf_prev_.value() + 0.05f * density_curr;
  this->density_lpf_prev_ = density_lpf;
  this->metrics.density.emplace_back(density_lpf);

}

void dlio::OdomNode::computeConvexHull() {

  // at least 4 keyframes for convex hull
  if (this->num_processed_keyframes < 4) {
    return;
  }

  // Reuse previous result when keyframe set is unchanged.
  if (this->convex_hull_last_kf_count_ == this->num_processed_keyframes && !this->keyframe_convex.empty()) {
    return;
  }

  // create a pointcloud with points at keyframes
  pcl::PointCloud<PointType>::Ptr cloud = std::make_shared<pcl::PointCloud<PointType>>();

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    PointType pt;
    pt.x = this->keyframes[i].first.first[0];
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
    cloud->emplace_back(pt);
  }
  lock.unlock();

  // calculate the convex hull of the point cloud
  this->convex_hull.setInputCloud(cloud);

  // get the indices of the keyframes on the convex hull
  pcl::PointCloud<PointType>::Ptr convex_points = std::make_shared<pcl::PointCloud<PointType>>();
  this->convex_hull.reconstruct(*convex_points);

  pcl::PointIndices::Ptr convex_hull_point_idx = std::make_shared<pcl::PointIndices>();
  this->convex_hull.getHullPointIndices(*convex_hull_point_idx);

  this->keyframe_convex.clear();
  for (int i=0; i<convex_hull_point_idx->indices.size(); ++i) {
    this->keyframe_convex.emplace_back(convex_hull_point_idx->indices[i]);
  }
  this->convex_hull_last_kf_count_ = this->num_processed_keyframes;

}

void dlio::OdomNode::computeConcaveHull() {

  // at least 5 keyframes for concave hull
  if (this->num_processed_keyframes < 5) {
    return;
  }

  const float alpha_curr = static_cast<float>(this->keyframe_thresh_dist_);
  constexpr float kAlphaRecomputeEps = 1e-3f;
  const bool alpha_changed = !this->concave_hull_last_alpha_ ||
                             std::abs(this->concave_hull_last_alpha_.value() - alpha_curr) > kAlphaRecomputeEps;

  // Recompute only when keyframe set grows or alpha changes enough.
  if (this->concave_hull_last_kf_count_ == this->num_processed_keyframes &&
      !alpha_changed &&
      !this->keyframe_concave.empty()) {
    return;
  }

  // create a pointcloud with points at keyframes
  auto cloud = std::make_shared<pcl::PointCloud<PointType>>();

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    PointType pt;
    pt.x = this->keyframes[i].first.first[0];
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
    cloud->emplace_back(pt);
  }
  lock.unlock();

  // calculate the concave hull of the point cloud
  this->concave_hull.setInputCloud(cloud);

  // get the indices of the keyframes on the concave hull
  pcl::PointCloud<PointType>::Ptr concave_points = std::make_shared<pcl::PointCloud<PointType>>();
  this->concave_hull.reconstruct(*concave_points);

  pcl::PointIndices::Ptr concave_hull_point_idx = std::make_shared<pcl::PointIndices>();
  this->concave_hull.getHullPointIndices(*concave_hull_point_idx);

  this->keyframe_concave.clear();
  for (int i=0; i<concave_hull_point_idx->indices.size(); ++i) {
    this->keyframe_concave.emplace_back(concave_hull_point_idx->indices[i]);
  }
  this->concave_hull_last_kf_count_ = this->num_processed_keyframes;
  this->concave_hull_last_alpha_ = alpha_curr;

}

void dlio::OdomNode::updateKeyframes() {

  // calculate difference in pose and rotation to all poses in trajectory
  float closest_d = std::numeric_limits<float>::infinity();
  int closest_idx = 0;
  int keyframes_idx = 0;

  int num_nearby = 0;

  for (const auto& k : this->keyframes) {

    // calculate distance between current pose and pose in keyframes
    float delta_d = sqrt( pow(this->state.p[0] - k.first.first[0], 2) +
                          pow(this->state.p[1] - k.first.first[1], 2) +
                          pow(this->state.p[2] - k.first.first[2], 2) );

    // count the number nearby current pose
    if (delta_d <= this->keyframe_thresh_dist_ * 1.5){
      ++num_nearby;
    }

    // store into variable
    if (delta_d < closest_d) {
      closest_d = delta_d;
      closest_idx = keyframes_idx;
    }

    keyframes_idx++;

  }

  // get closest pose and corresponding rotation
  Eigen::Vector3f closest_pose = this->keyframes[closest_idx].first.first;
  Eigen::Quaternionf closest_pose_r = this->keyframes[closest_idx].first.second;

  // calculate distance between current pose and closest pose from above
  float dd = sqrt( pow(this->state.p[0] - closest_pose[0], 2) +
                   pow(this->state.p[1] - closest_pose[1], 2) +
                   pow(this->state.p[2] - closest_pose[2], 2) );

  // calculate difference in orientation using SLERP
  Eigen::Quaternionf dq;

  if (this->state.q.dot(closest_pose_r) < 0.) {
    Eigen::Quaternionf lq = closest_pose_r;
    lq.w() *= -1.; lq.x() *= -1.; lq.y() *= -1.; lq.z() *= -1.;
    dq = this->state.q * lq.inverse();
  } else {
    dq = this->state.q * closest_pose_r.inverse();
  }

  double theta_rad = 2. * atan2(sqrt( pow(dq.x(), 2) + pow(dq.y(), 2) + pow(dq.z(), 2) ), dq.w());
  double theta_deg = theta_rad * (180.0/M_PI);

  // update keyframes
  bool newKeyframe = false;

  if (abs(dd) > this->keyframe_thresh_dist_ || abs(theta_deg) > this->keyframe_thresh_rot_) {
    newKeyframe = true;
  }

  if (abs(dd) <= this->keyframe_thresh_dist_) {
    newKeyframe = false;
  }

  if (abs(dd) <= this->keyframe_thresh_dist_ && abs(theta_deg) > this->keyframe_thresh_rot_ && num_nearby <= 1) {
    newKeyframe = true;
  }

  if (newKeyframe) {

    // update keyframe vector
    std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
    this->keyframes.emplace_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan));
    this->keyframe_timestamps.emplace_back(this->scan_header_stamp);
    this->keyframe_normals.emplace_back(this->gicp.getSourceCovariances());
    this->keyframe_transformations.emplace_back(this->T_corr);
    lock.unlock();
  }
}

void dlio::OdomNode::setAdaptiveParams() {

  // Spaciousness
  float sp = 0.f;
  float den = 0.f;
  {
    std::lock_guard<std::mutex> lock(this->metrics_mutex);
    if (!this->metrics.spaciousness.empty()) {
      sp = this->metrics.spaciousness.back();
    }
    if (!this->metrics.density.empty()) {
      den = this->metrics.density.back();
    }
  }

  if (sp < 0.5) { sp = 0.5; }
  if (sp > 5.0) { sp = 5.0; }

  this->keyframe_thresh_dist_ = sp;

  // Density
  if (den < 0.5*this->gicp_max_corr_dist_) { den = 0.5*this->gicp_max_corr_dist_; }
  if (den > 2.0*this->gicp_max_corr_dist_) { den = 2.0*this->gicp_max_corr_dist_; }

  if (sp <= 0.5f) {
    den = 0.5f * this->gicp_max_corr_dist_;
  } else if (sp >= 5.0f) {
    den = 2.0f * this->gicp_max_corr_dist_;
  }

  this->gicp.setMaxCorrespondenceDistance(den);

  // Concave hull alpha
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);

}

void dlio::OdomNode::pushSubmapIndices(const std::vector<float>& dists, int k, const std::vector<int>& frames) {

  const std::size_t n = std::min(dists.size(), frames.size());
  if (n == 0 || k <= 0) { return; }

  if (static_cast<std::size_t>(k) >= n) {
    this->submap_kf_idx_curr.insert(this->submap_kf_idx_curr.end(), frames.begin(), frames.begin() + n);
    return;
  }

  // Copy once, then select the k-th smallest in linear average time.
  std::vector<float> dists_work(dists.begin(), dists.begin() + n);
  const auto kth_it = dists_work.begin() + (k - 1);
  std::nth_element(dists_work.begin(), kth_it, dists_work.end());
  const float kth_element = *kth_it;

  for (std::size_t i = 0; i < n; ++i) {
    if (dists[i] <= kth_element) {
      this->submap_kf_idx_curr.emplace_back(frames[i]);
    }
  }

}

void dlio::OdomNode::buildSubmap(State vehicle_state) {

  // clear vector of keyframe indices to use for submap
  this->submap_kf_idx_curr.clear();

  // calculate distance between current pose and poses in keyframe set
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  std::vector<float> ds;
  std::vector<int> keyframe_nn;
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    float d = sqrt( pow(vehicle_state.p[0] - this->keyframes[i].first.first[0], 2) +
                    pow(vehicle_state.p[1] - this->keyframes[i].first.first[1], 2) +
                    pow(vehicle_state.p[2] - this->keyframes[i].first.first[2], 2) );
    ds.emplace_back(d);
    keyframe_nn.emplace_back(i);
  }
  lock.unlock();

  // get indices for top K nearest neighbor keyframe poses
  this->pushSubmapIndices(ds, this->submap_knn_, keyframe_nn);

  // get convex hull indices
  this->computeConvexHull();

  // get distances for each keyframe on convex hull
  std::vector<float> convex_ds;
  for (const auto& c : this->keyframe_convex) {
    convex_ds.emplace_back(ds[c]);
  }

  // get indices for top kNN for convex hull
  this->pushSubmapIndices(convex_ds, this->submap_kcv_, this->keyframe_convex);

  // get concave hull indices
  this->computeConcaveHull();

  // get distances for each keyframe on concave hull
  std::vector<float> concave_ds;
  for (const auto& c : this->keyframe_concave) {
    concave_ds.emplace_back(ds[c]);
  }

  // get indices for top kNN for concave hull
  this->pushSubmapIndices(concave_ds, this->submap_kcc_, this->keyframe_concave);

  // sort current and previous submap kf list of indices
  std::sort(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  std::sort(this->submap_kf_idx_prev.begin(), this->submap_kf_idx_prev.end());

  // remove duplicate indices
  auto last = std::unique(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  this->submap_kf_idx_curr.erase(last, this->submap_kf_idx_curr.end());

  // check if submap has changed from previous iteration
  if (this->submap_kf_idx_curr != this->submap_kf_idx_prev){

    this->submap_hasChanged = true;

    // Pause to prevent stealing resources from the main loop if it is running.
    this->pauseSubmapBuildIfNeeded();

    // reinitialize submap cloud and normals
    pcl::PointCloud<PointType>::Ptr submap_cloud_ = std::make_shared<pcl::PointCloud<PointType>>();
    std::shared_ptr<nano_gicp::CovarianceList> submap_normals_ (std::make_shared<nano_gicp::CovarianceList>());

    for (auto k : this->submap_kf_idx_curr) {

      // create current submap cloud
      lock.lock();
      *submap_cloud_ += *this->keyframes[k].second;
      lock.unlock();

      // grab corresponding submap cloud's normals
      submap_normals_->insert( std::end(*submap_normals_),
          std::begin(*(this->keyframe_normals[k])), std::end(*(this->keyframe_normals[k])) );
    }

    this->submap_cloud = submap_cloud_;
    this->submap_normals = submap_normals_;

    // Pause to prevent stealing resources from the main loop if it is running.
    this->pauseSubmapBuildIfNeeded();

    this->gicp_temp.setInputTarget(this->submap_cloud);
    this->submap_kdtree = this->gicp_temp.target_kdtree_;

    this->submap_kf_idx_prev = this->submap_kf_idx_curr;
  }
}

void dlio::OdomNode::buildKeyframesAndSubmap(State vehicle_state) {

  // transform the new keyframe(s) and associated covariance list(s)
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  const bool should_publish_keyframes =
      (this->kf_pose_pub->get_subscription_count() && this->kf_cloud_pub->get_subscription_count());
  std::vector<std::pair<std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                                  pcl::PointCloud<PointType>::ConstPtr>, rclcpp::Time>> pending_keyframes_to_publish;
  pending_keyframes_to_publish.reserve(this->keyframes.size() - this->num_processed_keyframes);

  for (int i = this->num_processed_keyframes; i < this->keyframes.size(); i++) {
    pcl::PointCloud<PointType>::ConstPtr raw_keyframe = this->keyframes[i].second;
    std::shared_ptr<const nano_gicp::CovarianceList> raw_covariances = this->keyframe_normals[i];
    Eigen::Matrix4f T = this->keyframe_transformations[i];
    lock.unlock();

    Eigen::Matrix4d Td = T.cast<double>();

    pcl::PointCloud<PointType>::Ptr transformed_keyframe = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::transformPointCloud (*raw_keyframe, *transformed_keyframe, T);

    std::shared_ptr<nano_gicp::CovarianceList> transformed_covariances (std::make_shared<nano_gicp::CovarianceList>(raw_covariances->size()));
    std::transform(raw_covariances->begin(), raw_covariances->end(), transformed_covariances->begin(),
                   [&Td](Eigen::Matrix4d cov) { return Td * cov * Td.transpose(); });

    ++this->num_processed_keyframes;

    lock.lock();
    this->keyframes[i].second = transformed_keyframe;
    this->keyframe_normals[i] = transformed_covariances;

    if (should_publish_keyframes) {
      pending_keyframes_to_publish.emplace_back(this->keyframes[i], this->keyframe_timestamps[i]);
    }
  }

  lock.unlock();

  // Publish newly processed keyframes in a single task to avoid per-keyframe scheduling overhead.
  if (should_publish_keyframes && !pending_keyframes_to_publish.empty()) {
    thread_pool.detach_task([this, pending_keyframes = std::move(pending_keyframes_to_publish)]() {
      for (const auto& item : pending_keyframes) {
        this->publishKeyframe(item.first, item.second);
      }
    });
  }

  // Pause to prevent stealing resources from the main loop if it is running.
  this->pauseSubmapBuildIfNeeded();

  this->buildSubmap(vehicle_state);
}

void dlio::OdomNode::pauseSubmapBuildIfNeeded() {
  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(this->main_loop_running_mutex);
  this->submap_build_cv.wait(lock, [this]{ return !this->main_loop_running; });
}

void dlio::OdomNode::debug() {
  // Total length traversed
  double length_traversed = 0.;
  Eigen::Vector3f p_curr = Eigen::Vector3f(0., 0., 0.);
  Eigen::Vector3f p_prev = Eigen::Vector3f(0., 0., 0.);
  for (const auto& t : this->trajectory) {
    if (p_prev == Eigen::Vector3f(0., 0., 0.)) {
      p_prev = t.first;
      continue;
    }
    p_curr = t.first;
    double l = sqrt(pow(p_curr[0] - p_prev[0], 2) + pow(p_curr[1] - p_prev[1], 2) + pow(p_curr[2] - p_prev[2], 2));

    if (l >= 0.1) {
      length_traversed += l;
      p_prev = p_curr;
    }
  }
  this->length_traversed = length_traversed;

  // Average computation time
  double avg_comp_time =
    std::accumulate(this->comp_times.begin(), this->comp_times.end(), 0.0) / this->comp_times.size();

  // Average sensor rates
  int win_size = 100;
  double avg_imu_rate;
  double avg_lidar_rate;
  if (this->imu_rates.size() < win_size) {
    avg_imu_rate =
      std::accumulate(this->imu_rates.begin(), this->imu_rates.end(), 0.0) / this->imu_rates.size();
  } else {
    avg_imu_rate =
      std::accumulate(this->imu_rates.end()-win_size, this->imu_rates.end(), 0.0) / win_size;
  }
  if (this->lidar_rates.size() < win_size) {
    avg_lidar_rate =
      std::accumulate(this->lidar_rates.begin(), this->lidar_rates.end(), 0.0) / this->lidar_rates.size();
  } else {
    avg_lidar_rate =
      std::accumulate(this->lidar_rates.end()-win_size, this->lidar_rates.end(), 0.0) / win_size;
  }

  // RAM Usage
  double vm_usage = 0.0;
  double resident_set = 0.0;
  std::ifstream stat_stream("/proc/self/stat", std::ios_base::in); //get info from proc directory
  std::string pid, comm, state, ppid, pgrp, session, tty_nr;
  std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  std::string utime, stime, cutime, cstime, priority, nice;
  std::string num_threads, itrealvalue, starttime;
  unsigned long vsize;
  long rss;
  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
              >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
              >> utime >> stime >> cutime >> cstime >> priority >> nice
              >> num_threads >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest
  stat_stream.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // for x86-64 is configured to use 2MB pages
  vm_usage = vsize / 1024.0;
  resident_set = rss * page_size_kb;

  // CPU Usage
  struct tms timeSample;
  clock_t now;
  double cpu_percent;
  now = times(&timeSample);
  if (now <= this->lastCPU || timeSample.tms_stime < this->lastSysCPU ||
      timeSample.tms_utime < this->lastUserCPU) {
      cpu_percent = -1.0;
  } else {
      cpu_percent = (timeSample.tms_stime - this->lastSysCPU) + (timeSample.tms_utime - this->lastUserCPU);
      cpu_percent /= (now - this->lastCPU);
      cpu_percent /= this->numProcessors;
      cpu_percent *= 100.;
  }
  this->lastCPU = now;
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;
  this->cpu_percents.emplace_back(cpu_percent);
  double avg_cpu_usage =
    std::accumulate(this->cpu_percents.begin(), this->cpu_percents.end(), 0.0) / this->cpu_percents.size();

  // Print to terminal
  printf("\033[2J\033[1;1H");

  std::cout << std::endl
            << "+-------------------------------------------------------------------+" << std::endl;
  std::cout << "|               Direct LiDAR-Inertial Odometry v" << this->version_  << "               |"
            << std::endl;
  std::cout << "+-------------------------------------------------------------------+" << std::endl;

  std::time_t curr_time = this->scan_stamp;
  std::string asc_time = std::asctime(std::localtime(&curr_time)); asc_time.pop_back();
  std::cout << "| " << std::left << asc_time;
  std::cout << std::right << std::setfill(' ') << std::setw(42)
    << "Elapsed Time: " + to_string_with_precision(this->elapsed_time, 2) + " seconds "
    << "|" << std::endl;

  if ( !this->cpu_type.empty() ) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << this->cpu_type + " x " + std::to_string(this->numProcessors)
      << "|" << std::endl;
  }

  if (this->sensor == dlio::SensorType::LIVOX) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Livox @ " + to_string_with_precision(avg_lidar_rate, 2)
                                  + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Unknown LiDAR @ " + to_string_with_precision(avg_lidar_rate, 2)
                                          + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  }

  std::cout << "|===================================================================|" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Position     {W}  [xyz] :: " + to_string_with_precision(this->state.p[0], 4) + " "
                                + to_string_with_precision(this->state.p[1], 4) + " "
                                + to_string_with_precision(this->state.p[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Orientation  {W} [wxyz] :: " + to_string_with_precision(this->state.q.w(), 4) + " "
                                + to_string_with_precision(this->state.q.x(), 4) + " "
                                + to_string_with_precision(this->state.q.y(), 4) + " "
                                + to_string_with_precision(this->state.q.z(), 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Lin Velocity {B}  [xyz] :: " + to_string_with_precision(this->state.v.lin.b[0], 4) + " "
                                + to_string_with_precision(this->state.v.lin.b[1], 4) + " "
                                + to_string_with_precision(this->state.v.lin.b[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Ang Velocity {B}  [xyz] :: " + to_string_with_precision(this->state.v.ang.b[0], 4) + " "
                                + to_string_with_precision(this->state.v.ang.b[1], 4) + " "
                                + to_string_with_precision(this->state.v.ang.b[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Accel Bias        [xyz] :: " + to_string_with_precision(this->state.b.accel[0], 8) + " "
                                + to_string_with_precision(this->state.b.accel[1], 8) + " "
                                + to_string_with_precision(this->state.b.accel[2], 8)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Gyro Bias         [xyz] :: " + to_string_with_precision(this->state.b.gyro[0], 8) + " "
                                + to_string_with_precision(this->state.b.gyro[1], 8) + " "
                                + to_string_with_precision(this->state.b.gyro[2], 8)
    << "|" << std::endl;

  std::cout << "|                                                                   |" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Distance Traveled  :: " + to_string_with_precision(length_traversed, 4) + " meters"
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Distance to Origin :: "
      + to_string_with_precision( sqrt(pow(this->state.p[0]-this->origin[0],2) +
                                       pow(this->state.p[1]-this->origin[1],2) +
                                       pow(this->state.p[2]-this->origin[2],2)), 4) + " meters"
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Registration       :: keyframes: " + std::to_string(this->keyframes.size()) + ", "
                               + "deskewed points: " + std::to_string(this->deskew_size)
    << "|" << std::endl;
  std::cout << "|                                                                   |" << std::endl;

  std::cout << std::right << std::setprecision(2) << std::fixed;
  std::cout << "| Computation Time :: "
    << std::setfill(' ') << std::setw(6) << this->comp_times.back()*1000. << " ms    // Avg: "
    << std::setw(6) << avg_comp_time*1000. << " / Max: "
    << std::setw(6) << *std::max_element(this->comp_times.begin(), this->comp_times.end())*1000.
    << "     |" << std::endl;
  std::cout << "| Cores Utilized   :: "
    << std::setfill(' ') << std::setw(6) << (cpu_percent/100.) * this->numProcessors << " cores // Avg: "
    << std::setw(6) << (avg_cpu_usage/100.) * this->numProcessors << " / Max: "
    << std::setw(6) << (*std::max_element(this->cpu_percents.begin(), this->cpu_percents.end()) / 100.)
                       * this->numProcessors
    << "     |" << std::endl;
  std::cout << "| CPU Load         :: "
    << std::setfill(' ') << std::setw(6) << cpu_percent << " %     // Avg: "
    << std::setw(6) << avg_cpu_usage << " / Max: "
    << std::setw(6) << *std::max_element(this->cpu_percents.begin(), this->cpu_percents.end())
    << "     |" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "RAM Allocation   :: " + to_string_with_precision(resident_set/1000., 2) + " MB"
    << "|" << std::endl;

  std::cout << "+-------------------------------------------------------------------+" << std::endl;

}
