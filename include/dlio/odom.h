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
#pragma once

#include "dlio/dlio.h"
#include "dlio/thread_pool.hpp"

// Equivariant IMU preintegration (Lie group-based, replaces manual quaternion integration)
#include "dlio/preintegration/preintegration.hpp"

// ROS
#include "rclcpp/rclcpp.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include "nav_msgs/msg/occupancy_grid.hpp"
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>

// STL
#include <atomic>
#include <optional>

// BOOST
#include <boost/format.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/adjacent_filtered.hpp>

// PCL
#include <pcl/filters/filter.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl_conversions/pcl_conversions.h>

#include "dlio/oct_vox_map.hpp"

// ── KeyframePoint: Point type for OctVoxMap that carries keyframe index ──
struct KeyframePoint {
  Eigen::Vector3f p;
  int idx = -1;

  float x() const { return p.x(); }
  float y() const { return p.y(); }
  float z() const { return p.z(); }
  float squaredNorm() const { return p.squaredNorm(); }
  auto array() const { return p.array(); }
};
inline KeyframePoint operator-(const KeyframePoint& a, const KeyframePoint& b) { return {a.p - b.p, -1}; }
inline KeyframePoint operator*(const KeyframePoint& a, float s)   { return {a.p * s, a.idx}; }
inline KeyframePoint operator+(const KeyframePoint& a, const KeyframePoint& b) { return {a.p + b.p, a.idx}; }
inline KeyframePoint operator/(const KeyframePoint& a, float s)   { return {a.p / s, a.idx}; }

class dlio::OdomNode: public rclcpp::Node {

public:

  OdomNode();
  ~OdomNode();

  void start();

private:

  struct State;
  struct ImuMeas;

  
  void getParams();

  void callbackPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr pc);
  void callbackImu(const sensor_msgs::msg::Imu::SharedPtr imu);

  void publishPose();

  void publishToROS(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud);
  void publishCloud(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud);
  void publishKeyframe(std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                       pcl::PointCloud<PointType>::ConstPtr> kf, rclcpp::Time timestamp);
  void publishKeyFrameOccupancyMap();
  void publishKeyFrameOccupancyMap(pcl::PointCloud<PointType>::Ptr cloud);
  void publishOccupancyMap(pcl::PointCloud<PointType>::Ptr cloud);

  void getScanFromROS(const sensor_msgs::msg::PointCloud2::SharedPtr& pc);
  void preprocessPoints();
  void deskewPointcloud();
  void initializeInputTarget();
  void setInputSource();

  void initializeDLIO();

  void getNextPose();
  bool imuMeasFromTimeRange(double start_time, double end_time,
                            boost::circular_buffer<ImuMeas>::reverse_iterator& begin_imu_it,
                            boost::circular_buffer<ImuMeas>::reverse_iterator& end_imu_it);
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
    integrateImu(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                 const std::vector<double>& sorted_timestamps);
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
    integrateImuInternal(Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                         const std::vector<double>& sorted_timestamps,
                         boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it,
                         boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it);
  void propagateGICP();

  void propagateState();
  void updateState();

  void setAdaptiveParams();

  void computeMetrics(const pcl::PointCloud<PointType>::ConstPtr& scan_snapshot, float density_snapshot);
  void computeSpaciousness(const pcl::PointCloud<PointType>::ConstPtr& scan_snapshot);
  void computeDensity(float density_curr);

  sensor_msgs::msg::Imu::SharedPtr transformImu(const sensor_msgs::msg::Imu::SharedPtr& imu);

  void updateKeyframes();
  void pushSubmapIndices(const std::vector<float>& dists, int k, const std::vector<int>& frames);
  void buildSubmap(State vehicle_state);
  void buildKeyframesAndSubmap(State vehicle_state);
  void pauseSubmapBuildIfNeeded();

  void debug();

  rclcpp::TimerBase::SharedPtr publish_timer;

  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
  rclcpp::CallbackGroup::SharedPtr lidar_cb_group, imu_cb_group;

  // Publishers
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr kf_pose_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr kf_cloud_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr deskewed_pub;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_map_pub;

  // TF
  std::shared_ptr<tf2_ros::TransformBroadcaster> br;

  // ROS Msgs
  nav_msgs::msg::Odometry odom_ros;
  geometry_msgs::msg::PoseStamped pose_ros;
  nav_msgs::msg::Path path_ros;
  geometry_msgs::msg::PoseArray kf_pose_ros;

  // Flags
  std::atomic<bool> dlio_initialized;
  std::atomic<bool> first_valid_scan;
  std::atomic<bool> first_imu_received;
  std::atomic<bool> imu_calibrated;
  std::atomic<bool> submap_hasChanged;
  std::atomic<bool> gicp_hasConverged;
  std::atomic<bool> deskew_status;
  std::atomic<bool> first_keyframe_occupancy_map;
  std::atomic<int> deskew_size;

  // Thread Pool
  BS::light_thread_pool thread_pool;
  // Equivariant IMU Preintegration (replaces manual quaternion integration)
  // Initialized after getParams() since gravity/noise values come from ROS params.
  using Pim = preintegration::EquivariantPreintegration<double>;
  std::shared_ptr<Pim::Params> pim_params_;
  std::optional<Pim> pim_;
  std::optional<Pim> pim_propagate_;         // dedicated pim for continuous state propagation

  // Per-LiDAR-frame propagation baseline (updated after each GICP correction)
  Eigen::Vector3d propagate_base_p_  = Eigen::Vector3d::Zero();
  Eigen::Quaterniond propagate_base_q_ = Eigen::Quaterniond::Identity();
  Eigen::Vector3d propagate_base_v_  = Eigen::Vector3d::Zero();

  // Trajectory
  std::vector<std::pair<Eigen::Vector3f, Eigen::Quaternionf>> trajectory;
  double length_traversed;
  std::optional<Eigen::Vector3f> length_ref_pose_;

  // Keyframes
  std::vector<std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                        pcl::PointCloud<PointType>::ConstPtr>> keyframes;
  std::vector<rclcpp::Time> keyframe_timestamps;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> keyframe_transformations;
  std::mutex keyframes_mutex;

  // Frames
  std::string odom_frame;
  std::string baselink_frame;
  std::string lidar_frame;
  std::string imu_frame;

  // Preprocessing
  pcl::CropBox<PointType> crop;
  pcl::VoxelGrid<PointType> voxel;

  // Point Clouds
  pcl::PointCloud<PointType>::ConstPtr original_scan;
  pcl::PointCloud<PointType>::ConstPtr deskewed_scan;
  pcl::PointCloud<PointType>::ConstPtr current_scan;

  // Keyframes
  int num_processed_keyframes;

  // OctVoxMap: O(1) incremental spatial voxel map,
  // replaces PCL ConvexHull/ConcaveHull for submap keyframe selection.
  using VoxelMap = dlio::OctVoxMap<KeyframePoint, float>;
  VoxelMap keyframe_voxel_map_;

  // Submap
  pcl::PointCloud<PointType>::ConstPtr submap_cloud;

  std::vector<int> submap_kf_idx_curr;
  std::vector<int> submap_kf_idx_prev;

  bool new_submap_is_ready;
  std::future<void> submap_future;
  std::condition_variable submap_build_cv;
  bool main_loop_running;
  std::mutex main_loop_running_mutex;

  // Timestamps
  rclcpp::Time scan_header_stamp;
  double scan_stamp;
  double prev_scan_stamp;
  boost::circular_buffer<double> comp_times;
  boost::circular_buffer<double> imu_rates;
  boost::circular_buffer<double> lidar_rates;

  double first_scan_stamp;
  double elapsed_time;

  // GICP
  small_gicp::RegistrationPCL<PointType, PointType> gicp;

  // Occupancy Map
  bool occupancy_enable;
  bool occupancy_use_dynamic_map;
  double occupancy_map_resolution;
  double occupancy_min_z;
  double occupancy_max_z;
  pcl::PointCloud<PointType>::Ptr occupancy_map_cloud_;

  // Transformations
  Eigen::Matrix4f T, T_prior, T_corr;

  Eigen::Vector3f origin;

  struct Extrinsics {
    struct SE3 {
      Eigen::Vector3f t;
      Eigen::Matrix3f R;
    };
    SE3 baselink2imu;
    SE3 baselink2lidar;
    Eigen::Matrix4f baselink2imu_T;
    Eigen::Matrix4f baselink2lidar_T;
  }; Extrinsics extrinsics;

  // IMU
  rclcpp::Time imu_stamp;
  double first_imu_stamp;
  double prev_imu_stamp;
  double prev_imu_stamp_for_transform_ = 0.;
  Eigen::Vector3f prev_ang_vel_cg_ = Eigen::Vector3f::Zero();

  struct ImuMeas {
    double stamp;
    double dt; // defined as the difference between the current and the previous measurement
    Eigen::Vector3f ang_vel;
    Eigen::Vector3f lin_accel;
  }; ImuMeas imu_meas;

  boost::circular_buffer<ImuMeas> imu_buffer;
  std::mutex mtx_imu;
  std::condition_variable cv_imu_stamp;
  int imu_calib_num_samples_;
  Eigen::Vector3f imu_calib_gyro_sum_;
  Eigen::Vector3f imu_calib_accel_sum_;
  bool imu_calib_print_once_;

  static bool comparatorImu(ImuMeas m1, ImuMeas m2) {
    return (m1.stamp < m2.stamp);
  };

  // Geometric Observer
  struct Geo {
    bool first_opt_done;
    std::mutex mtx;
    double dp;
    double dq_deg;
    Eigen::Vector3f prev_p;
    Eigen::Quaternionf prev_q;
    Eigen::Vector3f prev_vel;
  }; Geo geo;

  // State Vector
  struct ImuBias {
    Eigen::Vector3f gyro;
    Eigen::Vector3f accel;
  };

  struct Frames {
    Eigen::Vector3f b;
    Eigen::Vector3f w;
  };

  struct Velocity {
    Frames lin;
    Frames ang;
  };

  struct State {
    Eigen::Vector3f p; // position in world frame
    Eigen::Quaternionf q; // orientation in world frame
    Velocity v;
    ImuBias b; // imu biases in body frame
  }; State state;

  struct Pose {
    Eigen::Vector3f p; // position in world frame
    Eigen::Quaternionf q; // orientation in world frame
  };
  Pose lidarPose;

  // Metrics
  struct Metrics {
    std::vector<float> spaciousness;
    std::vector<float> density;
  }; Metrics metrics;
  std::mutex metrics_mutex;
  std::atomic<bool> metrics_task_running_{false};
  std::optional<float> spaciousness_lpf_prev_;
  std::optional<float> density_lpf_prev_;

  std::string cpu_type;
  boost::circular_buffer<double> cpu_percents;
  std::mutex debug_mutex;
  clock_t lastCPU, lastSysCPU, lastUserCPU;
  int numProcessors;

  // Parameters
  std::string version_;

  bool deskew_;

  double gravity_;

  bool time_offset_;

  bool adaptive_params_;

  bool debug_;

  double keyframe_thresh_dist_;
  double keyframe_thresh_rot_;

  int submap_knn_;
  int submap_kcv_;
  int submap_kcc_;
  int submap_voxel_radius_;
  int submap_kf_window_;

  bool densemap_filtered_;
  bool wait_until_move_;

  double crop_size_;

  bool vf_use_;
  double vf_res_;

  bool imu_calibrate_;
  bool calibrate_gyro_;
  bool calibrate_accel_;
  bool gravity_align_;
  double imu_calib_time_;
  int imu_buffer_size_;
  double imu_nominal_rate_;
  Eigen::Matrix3f imu_accel_sm_;

  int gicp_min_num_points_;
  int gicp_k_correspondences_;
  double gicp_max_corr_dist_;
  int gicp_max_iter_;
  double gicp_transformation_ep_;
  double gicp_rotation_ep_;
  bool gicp_require_converged_;
  int gicp_min_inliers_;
  double gicp_max_error_;
  bool gicp_reject_large_correction_;
  double gicp_max_corr_trans_;
  double gicp_max_corr_rot_deg_;

  bool degeneracy_enabled_;
  double degeneracy_eigen_thresh_;
  double degeneracy_soft_thresh_;

  double geo_Kp_;
  double geo_Kv_;
  double geo_Kq_;
  double geo_Kab_;
  double geo_Kgb_;
  double geo_abias_max_;
  double geo_gbias_max_;

};
