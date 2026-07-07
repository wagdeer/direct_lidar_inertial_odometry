/***********************************************************
 *                                                         *
 * DLIO MapEngine — loop closure & pose-graph backend     *
 *                                                         *
 * ROS-agnostic algorithmic core. Owned by OdomNode.      *
 * Direct function-call interface (no ROS topics).         *
 *                                                         *
 * Refactored from MapNode (2026-07-04)                    *
 *                                                         *
 ***********************************************************/

#pragma once

// ROS (used via node_ pointer for publishers/logging)
#include "rclcpp/rclcpp.hpp"
#include "direct_lidar_inertial_odometry/srv/save_pcd.hpp"

// Messages
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>

// nanoflann (DLIO built-in)
#include <nanoflann.hpp>

// GTSAM
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

// STL
#include <vector>
#include <deque>
#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <memory>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <functional>
#include <condition_variable>

namespace dlio {

class MapEngine {
public:

  // ── Types ──
  typedef pcl::PointXYZI PointType;
  typedef pcl::PointCloud<PointType> PointCloudType;
  typedef std::lock_guard<std::mutex> MtxLockGuard;
  typedef std::shared_ptr<Eigen::Affine3d> Affine3dPtr;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 3> KDTreeMatrix;
  typedef nanoflann::KDTreeEigenMatrixAdaptor<KDTreeMatrix, 3, nanoflann::metric_L2_Simple, true> KDTree;
  typedef nanoflann::RadiusResultSet<double, long int> RadiusResultSet;
  typedef std::pair<int, int> LoopEdgeID;

  /// Constructor. All ROS publishers/subscriptions/services are created on
  /// `node`. MapEngine does NOT own the node — OdomNode does.
  MapEngine(rclcpp::Node* node);
  ~MapEngine();

  // ── Per-frame callback ─────────────────────────────
  /// Called by OdomNode after each GICP correction.
  /// cloud: deskewed scan in odom frame (or base frame if mapped_cloud_=true)
  /// odom_pose: world→base transform from current odometry estimate
  /// stamp: scan timestamp
  void addFrame(const PointCloudType::ConstPtr& cloud,
                const Eigen::Affine3d& odom_pose,
                const rclcpp::Time& stamp);

  // ── Query optimized results ────────────────────────
  bool getLastOptimizedPose(Eigen::Affine3d& pose, int& id);
  size_t numOptimizedPoses() const;

  // ── Save ───────────────────────────────────────────
  bool isSaving() const { return saving_; }
  void savePCD(const std::string& path, float leaf_size);

private:

  // ── ROS node reference ─────────────────────────────
  rclcpp::Node* node_;

  // ── Publishers ─────────────────────────────────────
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_vis_graph_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_pgo_odom_;

  // ── Subscription / Service ─────────────────────────
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_save_req_;
  rclcpp::Service<direct_lidar_inertial_odometry::srv::SavePCD>::SharedPtr save_pcd_srv_;

  // ── Keyframe buffer ────────────────────────────────
  std::mutex mtx_buf_;
  std::deque<PointCloudType::Ptr> keyframes_cloud_;
  std::deque<Affine3dPtr> keyframes_odom_;
  std::deque<PointCloudType::Ptr> keyframes_cloud_copied_;
  std::deque<Affine3dPtr> keyframes_odom_copied_;
  std::deque<double> trajectory_dist_;
  std::deque<double> trajectory_dist_copied_;
  std::string odom_frame_id_;

  // ── ISAM2 / pose graph ─────────────────────────────
  gtsam::ISAM2 isam2_;
  std::mutex mtx_res_;
  gtsam::Values opt_result_;
  std::deque<LoopEdgeID> loop_edges_;
  gtsam::SharedNoiseModel prior_noise_, odom_noise_, const_loop_edge_noise_;
  size_t added_odom_id_, searched_loop_id_;

  // ── ICP / filtering ────────────────────────────────
  pcl::IterativeClosestPoint<PointType, PointType> icp_;
  pcl::VoxelGrid<PointType> vg_target_, vg_source_, vg_map_;

  // ── Threads ────────────────────────────────────────
  std::atomic<bool> stop_lc_thread_{false};
  std::atomic<bool> stop_viz_thread_{false};
  std::thread lc_thread_, viz_thread_, save_thread_;

  // ── Parameters ─────────────────────────────────────
  bool mapped_cloud_;
  double keyframe_dist_th_, keyframe_angular_dist_th_;
  double loop_search_time_diff_th_, loop_search_dist_diff_th_, loop_search_angular_dist_th_;
  int loop_search_frame_interval_;
  double search_radius_;
  int target_frame_num_;
  double target_voxel_leaf_size_, source_voxel_leaf_size_, vis_map_voxel_leaf_size_;
  double fitness_score_th_;
  int vis_map_cloud_frame_interval_;
  double leaf_size_;

  // ── Save state ─────────────────────────────────────
  std::atomic<bool> saving_{false};
  std::string save_directory_;

  // ── Viz colors ─────────────────────────────────────
  std_msgs::msg::ColorRGBA odom_edge_color_, loop_edge_color_, node_color_;
  double edge_scale_, node_scale_;

  // ── Methods ────────────────────────────────────────
  void getParams();
  void publishPgoOdom(const Eigen::Affine3d& affine, const nav_msgs::msg::Odometry& odom_msg);
  PointCloudType::Ptr buildMapCloud(int interval = 0);
  void publishMapCloud(const PointCloudType::Ptr& map);

  void copyKeyframes();
  void getLastPose(Eigen::Affine3d& pose, int& id);
  bool buildOdomGraph(gtsam::NonlinearFactorGraph& graph, gtsam::Values& init_estimate);
  bool buildKDTreeMat(KDTreeMatrix& mat);
  int searchTarget(const KDTree& kdtree, int id_query, const Eigen::Affine3d& pose_query, const rclcpp::Time& stamp_query);
  PointCloudType::Ptr buildTargetCloud(int target_id);
  PointCloudType::Ptr buildSourceCloud(int source_id);
  bool tryRegister(const Eigen::Affine3d& init_pose, const PointCloudType::Ptr& source,
                   const PointCloudType::Ptr& target, Eigen::Affine3d& result, double& score);
  bool buildLoopEdge(gtsam::NonlinearFactorGraph& graph);
  bool updateISAM2(const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& init_estimate);

  void visualizeLoop();
  void loopCloseLoop();

  // Vis helpers
  void buildVisOdomEdges(int n, const std_msgs::msg::Header& h, visualization_msgs::msg::Marker& m);
  void buildVisLoopEdges(int n, const std_msgs::msg::Header& h, visualization_msgs::msg::Marker& m);
  void buildVisNodes(int n, const std_msgs::msg::Header& h, visualization_msgs::msg::Marker& m);
  void publishVisGraph();

  // Save
  void saveCallback(const std_msgs::msg::String::ConstSharedPtr& dir);
  void saveFrames();
  void saveThreadLoop();

  // DLIO save_pcd service
  void savePCDService(
      std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Request> req,
      std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Response> res);
};

}  // namespace dlio
