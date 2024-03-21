#pragma once

#include "Rect.h"
#include "KalmanFilter.h"

#include "DataType.h"
#include "INIReader.h"

#include <cstddef>
#include <deque>
#include <memory>
#include <optional>

namespace ImprAssoc_track
{
enum class STrackState {
    New = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3,
};

class STrack
{
public:
    STrack(const Rect<float>& rect, const float& score,
        std::optional<FeatureVector> feat,
        int feat_history_size = 15) ;
    ~STrack();

    const Rect<float>& getRect() const;
    const STrackState& getSTrackState() const;

    const bool& isActivated() const;
    const float& getScore() const;
    const size_t& getTrackId() const;
    const size_t& getFrameId() const;
    const size_t& getStartFrameId() const;
    const size_t& getTrackletLength() const;

    void activate(const size_t& frame_id, const size_t& track_id);
    void reActivate(const STrack &new_track, const size_t &frame_id, const int &new_track_id = -1);

    void predict();
    void update(const STrack &new_track, const size_t &frame_id);

    void markAsLost();
    void markAsRemoved();

    void updateFeatures(const std::shared_ptr<FeatureVector> &feat);


public:
    std::shared_ptr<FeatureVector> curr_feat;
    std::unique_ptr<FeatureVector> smooth_feat;

private:
    KalmanFilter kalman_filter_;
    KalmanFilter::StateMean mean_;
    KalmanFilter::StateCov covariance_;

    Rect<float> rect_;
    STrackState state_;

    bool is_activated_;
    float score_;
    size_t track_id_;
    size_t frame_id_;
    size_t start_frame_id_;
    size_t tracklet_len_;

    static constexpr float alpha_ = 0.9;
    int feat_history_size_;
    std::deque<std::shared_ptr<FeatureVector>> feature_history_;

    void updateRect();
};
}