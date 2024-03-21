#pragma once

#include "STrack.h"
#include "lapjv.h"
#include "Object.h"
#include "ReID.h"

#include "DataType.h"
#include "INIReader.h"

#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <vector>

namespace ImprAssoc_track
{
class ImprAssocTracker
{
public:
    using STrackPtr = std::shared_ptr<STrack>;

    ImprAssocTracker(const int& frame_rate = 30,
                const int& track_buffer = 30,
                const std::string& reid_config_path = "config/reid.ini",
                const std::string& reid_onnx_model_path = "models/mobilenetv2_x1_4_msmt17.onnx",
                const float& high_thresh = 0.6,
                const float& low_thresh = 0.1,
                const float& match_thresh = 0.8,
                const float& new_track_thresh = 0.7,
                const float& second_match_thresh = 0.19,
                const float& overlap_thresh = 0.55,
                const float& iou_weight = 0.2,
                const float& proximity_thresh = 0.1,
                const float& appearance_thresh = 0.25,
                const bool& with_reid = true
                );

    ~ImprAssocTracker();

    std::vector<STrackPtr> update(const std::vector<Object>& objects, const cv::Mat& frame);

private:
    std::vector<STrackPtr> jointStracks(const std::vector<STrackPtr> &a_tlist,
                                        const std::vector<STrackPtr> &b_tlist) const;

    std::vector<STrackPtr> subStracks(const std::vector<STrackPtr> &a_tlist,
                                      const std::vector<STrackPtr> &b_tlist) const;

    void removeDuplicateStracks(const std::vector<STrackPtr> &a_stracks,
                                const std::vector<STrackPtr> &b_stracks,
                                std::vector<STrackPtr> &a_res,
                                std::vector<STrackPtr> &b_res) const;

    void linearAssignment(const std::vector<std::vector<float>> &cost_matrix,
                          const int &cost_matrix_size,
                          const int &cost_matrix_size_size,
                          const float &thresh,
                          std::vector<std::vector<int>> &matches,
                          std::vector<int> &b_unmatched,
                          std::vector<int> &a_unmatched) const;

    std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<bool>>>
    calcIouDistance(const std::vector<STrackPtr> &a_tracks,
                    const std::vector<STrackPtr> &b_tracks,
                    const float &thresh=0,
                    bool above=true,
                    bool distance = false) const;

    std::vector<std::vector<float>> fuseDistances(std::vector<std::vector<float>> &iou_costs,
                const std::vector<std::vector<float>> &embed_costs,
                const std::vector<std::vector<bool>> &iou_mask,
                const float &iou_weight) const;
 
    std::vector<std::vector<float>> calcIous(const std::vector<Rect<float>> &a_rect,
                                            const std::vector<Rect<float>> &b_rect) const;
    
    std::vector<std::vector<float>> calcDIous(const std::vector<Rect<float>> &a_rect,
                                            const std::vector<Rect<float>> &b_rect) const;

    std::vector<std::vector<float>> combineDists(const std::vector<std::vector<float>> &a_dists,
                        const std::vector<std::vector<float>> &b_dists) const;
    
    std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<bool>>>
    calcEmbeddingDistance(const std::vector<STrackPtr> &a_tracks,
                        const std::vector<STrackPtr> &b_tracks,
                        const float &max_embedding_distance,
                        const std::string &distance_metric) const;

    double execLapjv(const std::vector<std::vector<float> > &cost,
                     std::vector<int> &rowsol,
                     std::vector<int> &colsol,
                     bool extend_cost = false,
                     float cost_limit = std::numeric_limits<float>::max(),
                     bool return_cost = true) const;
    
    /**
     * @brief Calculate the cosine distance between two feature vectors
     * 
     * @param x Feature vector 1
     * @param y Feature vector 2
     * @return float Cosine distance (1 - cosine similarity)
     */
    float cosine_distance(const std::unique_ptr<FeatureVector> &x,
                                const std::shared_ptr<FeatureVector> &y) const
    {
        return 1.0f - (x->dot(*y) / (x->norm() * y->norm() + 1e-5f));
    }


    /**
     * @brief Calculate the euclidean distance between two feature vectors
     * 
     * @param x Feature vector 1
     * @param y Feature vector 2
     * @return float Euclidean distance
     */
    float euclidean_distance(const std::unique_ptr<FeatureVector> &x,
                                    const std::shared_ptr<FeatureVector> &y) const
    {
        return (x->transpose() - y->transpose()).norm();
    }

private:
    const float high_thresh_;
    const float match_thresh_;
    const float low_thresh_;
    const float new_track_thresh_;
    const float second_match_thresh_;
    const float overlap_thresh_;
    const float iou_weight_;
    const float proximity_thresh_;
    const float appearance_thresh_;
    const size_t max_time_lost_;
    std::unique_ptr<ReIDModel> reid_model_;
    bool reid_enabled_;

    size_t frame_id_;
    size_t track_id_count_;

    std::vector<STrackPtr> tracked_stracks_;
    std::vector<STrackPtr> lost_stracks_;
    std::vector<STrackPtr> removed_stracks_;

    FeatureVector _extract_features(const cv::Mat &frame,
                                         const Rect<float> &bbox_tlwh);

};
}