#pragma once
/// OctVoxMap — 八叉体素地图
///
/// Adapted from Super-LIO (RA-L 2026) for DLIO submap keyframe selection.
/// Replaces PCL ConvexHull/ConcaveHull with O(1) incremental voxel insertion.
///
/// Changes from Super-LIO version:
///   - tsl::robin_map → std::unordered_map (no external dependency)
///   - Removed PCL-dependent saveMap/resetMap
///   - Removed <execution>, <filesystem> includes
///   - Added radiusQueryKfIndices() for submap keyframe selection
///   - Namespace: LI2Sup → dlio

#include <set>
#include <list>
#include <queue>
#include <vector>
#include <memory>
#include <cstring>
#include <iostream>
#include <unordered_map>

#include <Eigen/Core>

#include "dlio/hknn_list60.h"

namespace dlio {

template<int K, typename Point>
class KNNHeap {
public:
  KNNHeap() : count(0), worst_(0), max_dist2_(0.0f) {
    memset(dist2_, 0, sizeof(dist2_));
  }

  void reset() {
    count = 0;
    worst_ = 0;
    max_dist2_ = 0.0f;
    memset(dist2_, 0, sizeof(dist2_));
  }

  uint8_t count;
  uint8_t worst_;
  float max_dist2_;
  float dist2_[K];
  std::array<Point, K> points_;

  inline void try_insert(float dist2, const Point& pt) {
    const bool not_full = (count < K);
    const bool should_insert = not_full || (dist2 < max_dist2_);
    
    if (should_insert) {
      const uint8_t insert_idx = not_full ? count : worst_;
      
      dist2_[insert_idx] = dist2;
      points_[insert_idx] = pt;
      
      if (not_full) {
        count++;
        if (dist2 > max_dist2_) {
          max_dist2_ = dist2;
          worst_ = insert_idx;
        }
      } else {
        update_worst_unrolled();
      }
    }
  }

private:
  inline void update_worst_unrolled() {
    float d0 = dist2_[0], d1 = dist2_[1], d2 = dist2_[2], d3 = dist2_[3], d4 = dist2_[4];
    
    uint8_t idx01 = d0 > d1 ? 0 : 1;
    float max01 = d0 > d1 ? d0 : d1;
    
    uint8_t idx23 = d2 > d3 ? 2 : 3;
    float max23 = d2 > d3 ? d2 : d3;
    
    uint8_t idx0123 = max01 > max23 ? idx01 : idx23;
    float max0123 = max01 > max23 ? max01 : max23;
    
    worst_ = max0123 > d4 ? idx0123 : 4;
    max_dist2_ = max0123 > d4 ? max0123 : d4;
  }

public:
  inline float max_dist2() const { return max_dist2_; }
};


template<typename Point>
class OctVox {
public:
  OctVox(const Point& pt, uint8_t local_idx)
  {
    counts_.fill(UNINIT_MASK);
    points_[local_idx] = pt;
    counts_[local_idx] = 1;
  }

  ~OctVox() {}

  void AddPoint(const Point& pt, uint8_t local_idx) {
    uint8_t& count = counts_[local_idx];
    Point& stored_point = points_[local_idx];
    if(count == UNINIT_MASK) {
      stored_point = pt;
      count = 1;
      return;
    }

    if(count >= MAX_POINTS_PER_SUBVOXEL) return;
    if ((pt - stored_point).squaredNorm() > DISTANCE_THRESHOLD_SQ) return;

    stored_point = (stored_point * count + pt) / (count + 1);
    ++count;
  }

  bool getPoint(const uint8_t local_idx, Point& pt) const {
    if (counts_[local_idx] == UNINIT_MASK) return false;
    pt = points_[local_idx];
    return true;
  }

  static constexpr uint8_t UNINIT_MASK = 0x00;
  static constexpr uint8_t MAX_POINTS_PER_SUBVOXEL = 20;
  static constexpr double DISTANCE_THRESHOLD_SQ = 0.1 * 0.1;

  std::array<uint8_t, 8> counts_;
  std::array<Point, 8> points_;
};



template<typename Point, typename Scalar>
class OctVoxMap {
public:
  using Ptr = std::shared_ptr<OctVoxMap>;
  using KEY = Eigen::Vector3i;
  using Points = std::vector<Point, Eigen::aligned_allocator<Point>>;
  using KNNHeapType = KNNHeap<5, Point>;
  using OctVoxType = OctVox<Point>;

  struct Options {
    float resolution      = 0.5;   
    std::size_t capacity  = 1000000;

    Options(float __resolution, std::size_t __capacity) {
      resolution = __resolution;
      capacity = __capacity;
    }
  };

  

  OctVoxMap() {
    flat_search_ptrs_.reserve(flat_search_order_offsets.size());
    for(std::size_t i = 0; i < flat_search_order_offsets.size(); i++){
      uint16_t start = flat_search_order_offsets[i];
      flat_search_ptrs_.push_back(const_cast<uint8_t*>(flat_search_order.data() + start));
    }
    group_idx_max_ = flat_search_order_offsets.size() - 1;
  }
  
  ~OctVoxMap() {
    grids_.clear();
    data_.clear();
  }
  
  OctVoxMap(Options options){
    SetOptions(options);
    flat_search_ptrs_.reserve(flat_search_order_offsets.size());
    for(std::size_t i = 0; i < flat_search_order_offsets.size(); i++){
      uint16_t start = flat_search_order_offsets[i];
      flat_search_ptrs_.push_back(const_cast<uint8_t*>(flat_search_order.data() + start));
    }
    group_idx_max_ = flat_search_order_offsets.size() - 1;
  }

  void SetOptions(const Options& options)
  {
    resolution_ = options.resolution;
    capacity_ = options.capacity;
    inv_resolution_ = 1.0 / resolution_;
    sub_resolution_ = resolution_ / 2.0;
    sub_inv_resolution_ = 1.0 / sub_resolution_;
  }

  void insert(const Points& cloud_world);
  void clear();
  void printInfo() const;

  /// Standard staged KNN search (kept for potential future P1 Staged KNN use).
  void getTopK(const Point& point, KNNHeapType& top_K) const;

  /// Voxel-neighbor KNN (kept for completeness).
  void getTopK_VN(const Point& point, KNNHeapType& top_K) const;

  /// Collect point indices from all voxels within |voxel_radius| (Chebyshev)
  /// of the voxel containing |center|. Designed for submap keyframe selection.
  /// Point must expose `.idx` (int) — satisfied by KeyframePoint.
  void radiusQueryKfIndices(const Eigen::Vector3f& center, int voxel_radius,
                            std::vector<int>& kf_indices) const;

  void reset_max_group(){
    group_idx_max_ = flat_search_order_offsets.size() - 1;
  }

  void decrease_max_group(){
    if(group_idx_max_ > 4) group_idx_max_--;
  }

  size_t size() const { return data_.size(); }

private:
  float resolution_ = 0.5;
  float inv_resolution_ = 1.0;
  float sub_resolution_ = 0.25;
  float sub_inv_resolution_ = 4.0;
  std::size_t capacity_ = 1000000;

  const KEY nearby_grids_[19] = {
    KEY(0, 0, 0),
    KEY(-1, -1, 0), KEY(-1, 0, 0), KEY(-1, 1, 0), 
    KEY(0, -1, 0), KEY(0, 1, 0), 
    KEY(1, -1, 0), KEY(1, 0, 0), KEY(1, 1, 0), 
    KEY(0, 0, -1), KEY(1, 0, -1), KEY(-1, 0, -1), 
    KEY(0, 1, -1), KEY(0, -1, -1), 
    KEY(0, 0, 1), KEY(1, 0, 1), KEY(-1, 0, 1), 
    KEY(0, 1, 1), KEY(0, -1, 1)
  };

  /// HashShiftMix for Eigen::Vector3i (needed for std::unordered_map).
  struct HASH_VEC {
    std::size_t operator()(const KEY &v) const {
      size_t h = static_cast<size_t>(v[0]);
      h ^= v[1] * 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= v[2] * 0x85ebca6b + (h << 6) + (h >> 2);
      return h;
    }
  };

  using DATA_LIST = std::list<std::pair<KEY, OctVoxType>>;
  using DATA_ITER = typename DATA_LIST::iterator;

  DATA_LIST data_;
  std::unordered_map<KEY, DATA_ITER, HASH_VEC> grids_;

  std::vector<uint8_t*> flat_search_ptrs_;
  int group_idx_max_;

};


// ── insert() ─────────────────────────────────────────

template<typename Point, typename Scalar>
void OctVoxMap<Point, Scalar>::insert(const Points& cloud_world){
  for(auto& pt : cloud_world){
    KEY fine_key = (pt * sub_inv_resolution_).array().floor().template cast<int>();
    KEY key;
    key[0] = fine_key[0] >> 1;
    key[1] = fine_key[1] >> 1;
    key[2] = fine_key[2] >> 1;

    uint8_t dx = fine_key[0] & 1;
    uint8_t dy = fine_key[1] & 1;
    uint8_t dz = fine_key[2] & 1;
    uint8_t local_idx = (dz << 2) | (dy << 1) | dx;

    auto iter = grids_.find(key);
    if (iter == grids_.end()) {
      data_.emplace_front(std::piecewise_construct,
        std::forward_as_tuple(key),
        std::forward_as_tuple(pt, local_idx));
      grids_.insert(std::make_pair(key, data_.begin()));
      
      if (data_.size() >= capacity_) {
        grids_.erase(data_.back().first);
        data_.pop_back();
      }
    } else {
      iter->second->second.AddPoint(pt, local_idx);
      data_.splice(data_.begin(), data_, iter->second);
    }
  }
}


// ── getTopK() — staged KNN ──────────────────────────

template<typename Point, typename Scalar>
void OctVoxMap<Point, Scalar>::getTopK(const Point& point, KNNHeapType& top_K) const {
  const KEY fine_key = (point * sub_inv_resolution_).array().floor().template cast<int>();
  KEY key;
  key[0] = fine_key[0] >> 1;
  key[1] = fine_key[1] >> 1;
  key[2] = fine_key[2] >> 1;

  const int dx = fine_key[0] & 1;
  const int dy = fine_key[1] & 1;
  const int dz = fine_key[2] & 1;
  const int local_idx = (dz << 2) | (dy << 1) | dx;
  const KEY mirror_axis = KEY(1 - (dx << 1), 1 - (dy << 1), 1 - (dz << 1));
  
  const int pre_voxel_ptr_size = 8;
  OctVoxType* top_voxels_2_search[pre_voxel_ptr_size];
  std::fill_n(top_voxels_2_search, pre_voxel_ptr_size, nullptr);
  
  for(uint8_t i = 0; i < pre_voxel_ptr_size; ++i)
  {
    KEY delta_key = mirror_axis.cwiseProduct(HKNN_neighbor_voxel[i]);
    KEY n_key = key + delta_key;
    if (auto iter = grids_.find(n_key); iter != grids_.end()) {
      top_voxels_2_search[i] = &iter->second->second;
    }
  }

  Point __sub_point;

  for (int group_idx = 0; group_idx < group_idx_max_; ++group_idx) {
    const uint8_t* group_it = flat_search_ptrs_[group_idx];
    const uint8_t* group_end = flat_search_ptrs_[group_idx + 1];

    while(group_it < group_end){
      const uint8_t neighbor_idx = *group_it++;
      uint8_t data_size = *group_it++;
      
      if(neighbor_idx < pre_voxel_ptr_size)
      {
        OctVoxType* voxel_ptr = top_voxels_2_search[neighbor_idx];
        if (voxel_ptr) {
          while (data_size--) {
            uint8_t _local_idx = (*group_it++)^local_idx;
            if (voxel_ptr->getPoint(_local_idx, __sub_point)) {
              const float dist2 = (__sub_point - point).squaredNorm();
              top_K.try_insert(dist2, __sub_point);
            }
          }
        }
        else group_it+=data_size;
        continue;
      }

      KEY delta_key = mirror_axis.cwiseProduct(HKNN_neighbor_voxel[neighbor_idx]);
      const KEY n_key = key + delta_key;

      if (auto iter = grids_.find(n_key); iter != grids_.end()){
        OctVoxType* voxel_ptr = &iter->second->second;
        while (data_size--){
          const uint8_t _local_idx = (*group_it++)^local_idx;
          if (voxel_ptr->getPoint(_local_idx, __sub_point)) {
            float dist2 = (__sub_point - point).squaredNorm();
            top_K.try_insert(dist2, __sub_point);
          }
        }
      }
      else group_it+=data_size;
    }

    if (top_K.count == 5)
      if (top_K.max_dist2_ < orders_min_dis2[group_idx]){
        break;
      }

  }
}


// ── getTopK_VN() — voxel-neighbor KNN ───────────────

template<typename Point, typename Scalar>
void OctVoxMap<Point, Scalar>::getTopK_VN(const Point& point, KNNHeapType& top_K) const{
  KEY key = (point * inv_resolution_).array().floor().template cast<int>();

  std::vector<OctVoxType*> voxels_2_search;
  voxels_2_search.reserve(19);
  for(std::size_t i = 0; i < 19; ++i) {
    KEY n_key = key + nearby_grids_[i];
    if (auto iter = grids_.find(n_key); iter != grids_.end()) {
      voxels_2_search.emplace_back(&iter->second->second);
    }
  }

  Point pt;
  for(auto& voxel : voxels_2_search) {
    for(uint8_t _i = 0; _i < 8; ++_i) {
      if(!voxel->getPoint(_i, pt)) continue;
      float dist2 = (pt - point).squaredNorm();
      top_K.try_insert(dist2, pt);
    }
  }
}


// ── radiusQueryKfIndices() ──────────────────────────
/// Collects keyframe indices from all occupied voxels within
/// voxel_radius (Chebyshev distance) of center's voxel.

template<typename Point, typename Scalar>
void OctVoxMap<Point, Scalar>::radiusQueryKfIndices(
    const Eigen::Vector3f& center, int voxel_radius,
    std::vector<int>& kf_indices) const
{
  const KEY center_key =
      (center * inv_resolution_).array().floor().template cast<int>();

  for (const auto& [key, iter] : grids_) {
    if ((key - center_key).array().abs().maxCoeff() <= voxel_radius) {
      const OctVoxType& voxel = iter->second;
      for (int i = 0; i < 8; ++i) {
        Point pt;
        if (voxel.getPoint(i, pt)) {
          kf_indices.push_back(pt.idx);
        }
      }
    }
  }
}


// ── clear() ──────────────────────────────────────────

template<typename Point, typename Scalar>
void OctVoxMap<Point, Scalar>::clear() {
  grids_.clear();
  data_.clear();
}


// ── printInfo() ──────────────────────────────────────

template<typename Point, typename Scalar>
void OctVoxMap<Point, Scalar>::printInfo() const {
    std::cout << " ---> OctVoxMap info. Size: " << data_.size() 
              << " Capacity: " << capacity_ << std::endl;
}

}  // namespace dlio
