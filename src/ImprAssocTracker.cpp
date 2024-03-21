#include "ImprAssocTrack/ImprAssocTracker.h"
#include "ImprAssocTrack/DataType.h"
#include "ImprAssocTrack/INIReader.h"
#include <opencv2/imgproc.hpp>


#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <iostream>


ImprAssoc_track::ImprAssocTracker::ImprAssocTracker(
                                    const int& frame_rate,
                                    const int& track_buffer,
                                    const std::string& reid_config_path,
                                    const std::string& reid_onnx_model_path,
                                    const float& high_thresh,
                                    const float& match_thresh,
                                    const float& low_thresh,
                                    const float& new_track_thresh,
                                    const float& second_match_thresh,
                                    const float& overlap_thresh,
                                    const float& iou_weight,
                                    const float& proximity_thresh,
                                    const float& appearance_thresh,
                                    const bool& with_reid
                                    ) :
    high_thresh_(high_thresh),
    low_thresh_(low_thresh),
    new_track_thresh_(new_track_thresh),
    second_match_thresh_(second_match_thresh),
    overlap_thresh_(overlap_thresh),
    iou_weight_(iou_weight),
    match_thresh_(match_thresh),
    proximity_thresh_(proximity_thresh),
    appearance_thresh_(appearance_thresh),
    max_time_lost_(static_cast<size_t>(frame_rate / 30.0 * track_buffer)),
    frame_id_(0),
    track_id_count_(0)
{
    if (with_reid && reid_config_path.size() > 0 && reid_onnx_model_path.size() > 0) {
        reid_model_ = std::make_unique<ReIDModel>(reid_config_path, reid_onnx_model_path);
        reid_enabled_ = true;
    } else {
        reid_enabled_ = false;
    }
}

ImprAssoc_track::ImprAssocTracker::~ImprAssocTracker()
{
}

std::vector<ImprAssoc_track::ImprAssocTracker::STrackPtr> ImprAssoc_track::ImprAssocTracker::update(const std::vector<Object>& objects, const cv::Mat& frame)
{
    ////////////////// Step 1: Get detections //////////////////
    frame_id_++;

    // Create new STracks using the result of object detection
    std::vector<STrackPtr> det_stracks;
    std::vector<STrackPtr> det_low_stracks;

    for (const auto &object : objects)
    {
        if (object.prob >= low_thresh_){
            std::shared_ptr<STrack> strack;

            // std::cout << "Before _extract_features" << std::endl;
            // std::cout.flush();

            if (reid_enabled_){
                FeatureVector embedding = _extract_features(frame, object.rect);
                // std::cout << "after _extract_features" << std::endl;
                // std::cout.flush();
                std::optional<FeatureVector> optional_embedding(embedding); // Wrap the embedding in an std::optional

                strack = std::make_shared<STrack>(object.rect, object.prob, optional_embedding, 15);
            } else {
                FeatureVector embedding;
                strack = std::make_shared<STrack>(object.rect, object.prob, embedding, 15);
            }

            if (object.prob >= high_thresh_)
            {
                std::cout   << "strack "
                            << strack->getScore()
                            << " into high det\n";
                det_stracks.push_back(strack);
            }
            else
            {
                det_low_stracks.push_back(strack);
            }
        }


    }

    // Create lists of existing STrack
    std::vector<STrackPtr> active_stracks;
    std::vector<STrackPtr> non_active_stracks;
    std::vector<STrackPtr> strack_pool;
    active_stracks.reserve(tracked_stracks_.size());
    non_active_stracks.reserve(tracked_stracks_.size());


    for (const auto& tracked_strack : tracked_stracks_)
    {
        if (!tracked_strack->isActivated())
        {
            std::cout   << "tracked strack "
                        << tracked_strack->getTrackId()
                        << " into non active stracks\n";
            std::cout.flush();
            non_active_stracks.push_back(tracked_strack);
        }
        else
        {
            std::cout   << "tracked strack "
                        << tracked_strack->getTrackId()
                        << " into active stracks\n";
            std::cout.flush();
            active_stracks.push_back(tracked_strack);
        }
    }

    strack_pool = jointStracks(active_stracks, lost_stracks_);

    // Predict current pose by KF
    for (const auto &strack : strack_pool)
    {
        strack->predict();
    }

    ////////////////// Step 2: First association, with IoU //////////////////
    std::vector<STrackPtr> current_tracked_stracks;
    std::vector<STrackPtr> remain_tracked_stracks;
    std::vector<STrackPtr> remain_det_stracks;
    std::vector<STrackPtr> refind_stracks;
    std::vector<STrackPtr> current_lost_stracks;


    {
        std::vector<std::vector<int>> matches_idx;
        std::vector<int> unmatch_detection_idx, unmatch_track_idx;


        std::vector<std::vector<float>> d_iou_dists, iou_dists,
                iou_dists_second, raw_emb_dists, second_raw_emb_dists;
        std::vector<std::vector<bool>> emb_dist_mask_1st_assoc,
                below_iou_thresh_mask, second_below_iou_thresh_mask,
                throw_away;
        
        // std::cout << "Before calcIouDistance (D)" << std::endl;
        // std::cout.flush();
        std::tie(d_iou_dists, throw_away) = calcIouDistance(strack_pool, det_stracks, 0, true, true);
        // std::cout << "After calcIouDistance (D)" << std::endl;
        // std::cout.flush();

        // Mask off ious below proximity_thresh_.
        std::tie(iou_dists, below_iou_thresh_mask) = calcIouDistance(strack_pool, det_stracks, proximity_thresh_, false, false);
        // std::cout << "After calcIouDistance" << std::endl;
        // std::cout.flush();
        // TODO: If we are using Re-ID then add in embeddings 
        if (reid_enabled_) {
            // embedding dists
            // std::cout << "Before calcEmbeddingDistance" << std::endl;
            // std::cout.flush();
            std::tie(raw_emb_dists, emb_dist_mask_1st_assoc) = 
                    calcEmbeddingDistance(strack_pool, det_stracks,
                                    appearance_thresh_,
                                    reid_model_->get_distance_metric());
            // std::cout << "After calcEmbeddingDistance" << std::endl;
            // std::cout.flush();
        }

        const auto dists_first = fuseDistances(d_iou_dists, raw_emb_dists, below_iou_thresh_mask, iou_weight_);
        // std::cout << "After fuseDistances high" << std::endl;
        // std::cout.flush();
        // Mask off ious below 1-second_match_thresh.
        // std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<bool>>>
        std::tie(iou_dists_second, second_below_iou_thresh_mask) = calcIouDistance(strack_pool, det_low_stracks, 1-second_match_thresh_, false, false);
        // std::cout << "After calcIouDistance (low)" << std::endl;
        // std::cout.flush();
        const auto dists_second = fuseDistances(iou_dists_second, second_raw_emb_dists, second_below_iou_thresh_mask, 1);
        
        // std::cout << "After fuseDistances low" << std::endl;
        // std::cout.flush();

        const float B = match_thresh_ / second_match_thresh_;

        for (auto& dist_row : dists_second){
            for (auto dist : dist_row){
                dist *= B;
            }
        }

        // concat dists
        std::vector<std::vector<float>> combined_dists = combineDists(d_iou_dists, dists_second);

        // concat dets
        det_stracks.insert( det_stracks.end(), det_low_stracks.begin(), det_low_stracks.end() );
        std::cout   << "size of det stracks "
                    << det_stracks.size()
                    << "\n";
        std::cout.flush();

        linearAssignment(combined_dists, strack_pool.size(), det_stracks.size(), match_thresh_,
                         matches_idx, unmatch_track_idx, unmatch_detection_idx);

        for (const auto &match_idx : matches_idx)
        {
            const auto track = strack_pool[match_idx[0]];
            const auto det = det_stracks[match_idx[1]];
            if (track->getSTrackState() == STrackState::Tracked)
            {
                track->update(*det, frame_id_);
                current_tracked_stracks.push_back(track);
            }
            else
            {
                track->reActivate(*det, frame_id_);
                refind_stracks.push_back(track);
            }
        }

        for (const auto &unmatch_idx : unmatch_detection_idx)
        {
            std::cout   << "add det "
                        << unmatch_idx
                        << " to remain dets\n";
            std::cout.flush();
            std::cout   << "det "
                        << det_stracks[unmatch_idx]->getScore()
                        << " about to be added to remain_det_stracks\n";
            std::cout.flush();
            remain_det_stracks.push_back(det_stracks[unmatch_idx]);
        }

        for (const auto &unmatch_track : unmatch_track_idx)
        {
            const auto track = strack_pool[unmatch_track];
            if (track->getSTrackState() != STrackState::Lost)
            {
                track->markAsLost();
                current_lost_stracks.push_back(track);
            }
        }
    }

    ////////////////// Step 4: Init new stracks //////////////////
    //////////////////////////// OAI /////////////////////////////

    // First calculate the iou between every unmatched det and all tracks. If the max iou
    // for a det D is above overlap_thresh, discard it.
    std::vector<STrackPtr> current_removed_stracks;

    {
        std::vector<int> unmatch_detection_idx;
        std::vector<int> unmatch_unconfirmed_idx;
        std::vector<std::vector<int>> matches_idx;
        std::vector<std::vector<float>> unmatched_iou_costs;
        std::vector<std::vector<bool>> throw_away;

        // Calc iou dists (1-iou)
        std::tie(unmatched_iou_costs, throw_away) = calcIouDistance(remain_det_stracks, strack_pool, 0, true, false);
        // linearAssignment(dists, non_active_stracks.size(), remain_det_stracks.size(), 0.7,
        //                  matches_idx, unmatch_unconfirmed_idx, unmatch_detection_idx);

        // loop over the detections
        std::cout   << "size of remain dets "
                    << remain_det_stracks.size()
                    << "\n";
        std::cout   << "size of strack pool "
                    << strack_pool.size()
                    << "\n";
        std::cout.flush();
        for (auto i=0; i < remain_det_stracks.size(); i++){
            if (strack_pool.size() > 0){
                // minimum is the lowest cost (1-iou) and highest iou
                auto minimum = std::min_element(unmatched_iou_costs[i].begin(), unmatched_iou_costs[i].end());
                // 1 - min to get iou
                if ((1 - *minimum) < overlap_thresh_){
                    const auto track = remain_det_stracks[i];
                    if (track->getScore() > new_track_thresh_) {
                        track_id_count_++;
                        track->activate(frame_id_, track_id_count_);
                        if (reid_enabled_) {
                            FeatureVector embedding = _extract_features(frame, track->getRect());
                            std::shared_ptr<FeatureVector> embedding_ptr = std::make_shared<FeatureVector>(embedding);
                            track->updateFeatures(embedding_ptr);
                        }
                        current_tracked_stracks.push_back(track);
                    }
                }
            } else {
                const auto track = remain_det_stracks[i];
                std::cout   << "det "
                            << track->getScore()
                            << " about to be activated\n";
                std::cout.flush();

                if (track->getScore() > new_track_thresh_) {
                    track_id_count_++;
                    std::cout   << "activate det "
                                << track_id_count_
                                << "\n";
                    std::cout.flush();
                    track->activate(frame_id_, track_id_count_);
                    if (reid_enabled_) {
                        FeatureVector embedding = _extract_features(frame, track->getRect());
                        std::shared_ptr<FeatureVector> embedding_ptr = std::make_shared<FeatureVector>(embedding);
                        track->updateFeatures(embedding_ptr);
                    }
                    std::cout   << "size of current tracked stracks "
                                << current_tracked_stracks.size()
                                << "\n";
                    std::cout.flush();
                    current_tracked_stracks.push_back(track);
                }
            }
        }
    }

    ////////////////// Step 5: Update state //////////////////
    for (const auto &lost_strack : lost_stracks_)
    {
        if (frame_id_ - lost_strack->getFrameId() > max_time_lost_)
        {
            lost_strack->markAsRemoved();
            current_removed_stracks.push_back(lost_strack);
        }
    }

    tracked_stracks_ = jointStracks(current_tracked_stracks, refind_stracks);
    lost_stracks_ = subStracks(jointStracks(subStracks(lost_stracks_, tracked_stracks_), current_lost_stracks), removed_stracks_);
    removed_stracks_ = jointStracks(removed_stracks_, current_removed_stracks);

    std::vector<STrackPtr> tracked_stracks_out, lost_stracks_out;
    removeDuplicateStracks(tracked_stracks_, lost_stracks_, tracked_stracks_out, lost_stracks_out);
    tracked_stracks_ = tracked_stracks_out;
    lost_stracks_ = lost_stracks_out;

    std::vector<STrackPtr> output_stracks;
    for (const auto &track : tracked_stracks_)
    {
        if (track->isActivated())
        {
            output_stracks.push_back(track);
        }
    }

    return output_stracks;
}
std::vector<ImprAssoc_track::ImprAssocTracker::STrackPtr> ImprAssoc_track::ImprAssocTracker::jointStracks(const std::vector<STrackPtr> &a_tlist,
                                                                                      const std::vector<STrackPtr> &b_tlist) const
{
    std::map<int, int> exists;
    std::vector<STrackPtr> res;
    for (size_t i = 0; i < a_tlist.size(); i++)
    {
        exists.emplace(a_tlist[i]->getTrackId(), 1);
        res.push_back(a_tlist[i]);
    }
    for (size_t i = 0; i < b_tlist.size(); i++)
    {
        const int &tid = b_tlist[i]->getTrackId();
        if (!exists[tid] || exists.count(tid) == 0)
        {
            exists[tid] = 1;
            res.push_back(b_tlist[i]);
        }
    }
    return res;
}

std::vector<ImprAssoc_track::ImprAssocTracker::STrackPtr> ImprAssoc_track::ImprAssocTracker::subStracks(const std::vector<STrackPtr> &a_tlist,
                                                                                    const std::vector<STrackPtr> &b_tlist) const
{
    std::map<int, STrackPtr> stracks;
    for (size_t i = 0; i < a_tlist.size(); i++)
    {
        stracks.emplace(a_tlist[i]->getTrackId(), a_tlist[i]);
    }

    for (size_t i = 0; i < b_tlist.size(); i++)
    {
        const int &tid = b_tlist[i]->getTrackId();
        if (stracks.count(tid) != 0)
        {
            stracks.erase(tid);
        }
    }

    std::vector<STrackPtr> res;
    std::map<int, STrackPtr>::iterator it;
    for (it = stracks.begin(); it != stracks.end(); ++it)
    {
        res.push_back(it->second);
    }

    return res;
}

void ImprAssoc_track::ImprAssocTracker::removeDuplicateStracks(const std::vector<STrackPtr> &a_stracks,
                                                     const std::vector<STrackPtr> &b_stracks,
                                                     std::vector<STrackPtr> &a_res,
                                                     std::vector<STrackPtr> &b_res) const
{
    std::vector<std::vector<float>> ious;
    std::vector<std::vector<bool>> throw_away;
    std::tie(ious, throw_away) = calcIouDistance(a_stracks, b_stracks, 0, false, false);

    std::vector<std::pair<size_t, size_t>> overlapping_combinations;
    for (size_t i = 0; i < ious.size(); i++)
    {
        for (size_t j = 0; j < ious[i].size(); j++)
        {
            if (ious[i][j] < 0.15)
            {
                overlapping_combinations.emplace_back(i, j);
            }
        }
    }

    std::vector<bool> a_overlapping(a_stracks.size(), false), b_overlapping(b_stracks.size(), false);
    for (const auto &[a_idx, b_idx] : overlapping_combinations)
    {
        const int timep = a_stracks[a_idx]->getFrameId() - a_stracks[a_idx]->getStartFrameId();
        const int timeq = b_stracks[b_idx]->getFrameId() - b_stracks[b_idx]->getStartFrameId();
        if (timep > timeq)
        {
            b_overlapping[b_idx] = true;
        }
        else
        {
            a_overlapping[a_idx] = true;
        }
    }

    for (size_t ai = 0; ai < a_stracks.size(); ai++)
    {
        if (!a_overlapping[ai])
        {
            a_res.push_back(a_stracks[ai]);
        }
    }

    for (size_t bi = 0; bi < b_stracks.size(); bi++)
    {
        if (!b_overlapping[bi])
        {
            b_res.push_back(b_stracks[bi]);
        }
    }
}

void ImprAssoc_track::ImprAssocTracker::linearAssignment(const std::vector<std::vector<float>> &cost_matrix,
                                               const int &cost_matrix_size,
                                               const int &cost_matrix_size_size,
                                               const float &thresh,
                                               std::vector<std::vector<int>> &matches,
                                               std::vector<int> &a_unmatched,
                                               std::vector<int> &b_unmatched) const
{
    if (cost_matrix.size() == 0)
    {
        for (int i = 0; i < cost_matrix_size; i++)
        {
            // std::cout   << "add "
            //             << i
            //             << " to tracks unmatched\n";
            a_unmatched.push_back(i);
        }
        for (int i = 0; i < cost_matrix_size_size; i++)
        {
            // std::cout   << "add "
            //             << i
            //             << " to dets unmatched\n";
            b_unmatched.push_back(i);
        }
        return;
    }

    std::vector<int> rowsol; std::vector<int> colsol;
    execLapjv(cost_matrix, rowsol, colsol, true, thresh);
    for (size_t i = 0; i < rowsol.size(); i++)
    {
        if (rowsol[i] >= 0)
        {
            std::vector<int> match;
            match.push_back(i);
            match.push_back(rowsol[i]);
            matches.push_back(match);
        }
        else
        {
            a_unmatched.push_back(i);
        }
    }

    for (size_t i = 0; i < colsol.size(); i++)
    {
        if (colsol[i] < 0)
        {
            b_unmatched.push_back(i);
        }
    }
}

std::vector<std::vector<float>> ImprAssoc_track::ImprAssocTracker::combineDists(const std::vector<std::vector<float>> &a_dists,
                                                                  const std::vector<std::vector<float>> &b_dists) const
{
    std::vector<std::vector<float>> combined_dists;
    for (auto i = 0; i < a_dists.size(); i++)
    {
        std::vector<float> cost = a_dists[i];
        // Dets in b.
        if (b_dists.size()!=0){
            for (auto j = 0; j < b_dists[i].size(); j++)
            {
                cost.push_back(b_dists[i][j]);
            }
        }
        combined_dists.push_back(cost);
    }
    return combined_dists;
}

FeatureVector ImprAssoc_track::ImprAssocTracker::_extract_features(const cv::Mat &frame,
                                         const Rect<float> &bbox_tlwh)
{
    // std::cout << "In _extract_features" << std::endl;
    // std::cout.flush();

    cv::Rect_<float> cv_bbox;
    cv_bbox.x = bbox_tlwh.x();
    cv_bbox.y = bbox_tlwh.y();
    cv_bbox.height = bbox_tlwh.height();
    cv_bbox.width = bbox_tlwh.width();

    if (frame.empty()) {
        std::cout << "Frame is empty" << std::endl;
        FeatureVector features;
        return features;
    }

    if (cv_bbox.x >= 0 && cv_bbox.y >= 0 &&
        cv_bbox.x + cv_bbox.width <= frame.cols &&
        cv_bbox.y + cv_bbox.height <= frame.rows) {
        // Safe to extract the patch
        // std::cout << "Start setting up patch" << std::endl;
        // std::cout.flush();
        cv::Mat patch = frame(cv_bbox);
        // std::cout << "Finished setting up patch" << std::endl;
        // std::cout.flush();
        return reid_model_->extract_features(patch);
    } else {
        std::cout << "The box does not fit in the frame" << std::endl;
    }


}

std::vector<std::vector<float>>
ImprAssoc_track::ImprAssocTracker::calcIous(const std::vector<Rect<float>> &a_rect,
                                                                  const std::vector<Rect<float>> &b_rect) const
{
    std::vector<std::vector<float>> ious;
    if (a_rect.size() * b_rect.size() == 0)
    {
        return ious;
    }

    ious.resize(a_rect.size());
    for (size_t i = 0; i < ious.size(); i++)
    {
        ious[i].resize(b_rect.size());
    }

    for (size_t bi = 0; bi < b_rect.size(); bi++)
    {
        for (size_t ai = 0; ai < a_rect.size(); ai++)
        {
            const float iou = b_rect[bi].calcIoU(a_rect[ai]);

            ious[ai][bi] = iou;

        }
    }
    return ious;
}
/*TODO: Need to implement DIou dists*/
std::vector<std::vector<float>>
ImprAssoc_track::ImprAssocTracker::calcDIous(const std::vector<Rect<float>> &a_rect,
                                                                  const std::vector<Rect<float>> &b_rect) const
{
    std::vector<std::vector<float>> ious;
    if (a_rect.size() * b_rect.size() == 0)
    {
        return ious;
    }

    ious.resize(a_rect.size());
    for (size_t i = 0; i < ious.size(); i++)
    {
        ious[i].resize(b_rect.size());
    }

    for (size_t bi = 0; bi < b_rect.size(); bi++)
    {
        for (size_t ai = 0; ai < a_rect.size(); ai++)
        {
            const float iou = b_rect[bi].calcIoU(a_rect[ai]);
            const float R = b_rect[bi].calcR(a_rect[ai]);
            const float d_iou = iou + R;

            ious[ai][bi] = d_iou;
              
        }
    }
    return ious;
}

std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<bool>>>
ImprAssoc_track::ImprAssocTracker::calcIouDistance(const std::vector<STrackPtr> &a_tracks,
                                                    const std::vector<STrackPtr> &b_tracks,
                                                    const float &thresh,
                                                    const bool above,
                                                    const bool distance) const
{
    std::vector<ImprAssoc_track::Rect<float>> a_rects, b_rects;
    for (size_t i = 0; i < a_tracks.size(); i++)
    {
        a_rects.push_back(a_tracks[i]->getRect());
    }

    for (size_t i = 0; i < b_tracks.size(); i++)
    {
        b_rects.push_back(b_tracks[i]->getRect());
    }
    std::vector<std::vector<float>> ious;
    if (distance) {
        ious = calcDIous(a_rects, b_rects);
    }
    else {
        ious = calcIous(a_rects, b_rects);
    }

    std::vector<std::vector<float>> cost_matrix;
    std::vector<std::vector<bool>> mask;
    mask.resize(ious.size());
    for (size_t i = 0; i < ious.size(); i++)
    {
        mask[i].resize(ious[i].size());

        std::vector<float> iou;
        for (size_t j = 0; j < ious[i].size(); j++)
        {
            if (ious[i][j] < thresh) {
                mask[i][j] = false;
            } else {
                mask[i][j] = true;
            }
            iou.push_back(1 - ious[i][j]);
        }
        cost_matrix.push_back(iou);
    }

    return {cost_matrix, mask};
}

std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<bool>>>
ImprAssoc_track::ImprAssocTracker::calcEmbeddingDistance(const std::vector<STrackPtr> &a_tracks,
                                                                          const std::vector<STrackPtr> &b_tracks,
                                                                          const float &max_embedding_distance,
                                                                          const std::string &distance_metric) const
{
    if (!(distance_metric == "euclidean" || distance_metric == "cosine")) {
        std::cout << "Invalid distance metric " << distance_metric
                  << " passed.";
        std::cout << "Only 'euclidean' and 'cosine' are supported."
                  << std::endl;
        exit(1);
    }
    // std::cout << "In calcEmbeddingDistance" << std::endl;
    // std::cout.flush();

    size_t num_tracks = a_tracks.size();
    size_t num_detections = b_tracks.size();

    std::vector<std::vector<float>> cost_matrix;
    std::vector<std::vector<bool>> mask;

    if (num_tracks * num_detections == 0){
        // std::cout << "empty cost matrix" << std::endl;
        // std::cout.flush();
        return {cost_matrix, mask};
    }

    cost_matrix.resize(num_tracks);
    mask.resize(num_tracks);
    for (int i=0; i < num_tracks; i++) {
        cost_matrix[i].resize(num_detections);
        mask[i].resize(num_detections);
        for (int j=0; j < num_detections; j++){
            float dist;

            // std::cout << "Before distance calcs: " << distance_metric << std::endl;
            // std::cout.flush();

            if (distance_metric == "euclidean") {
                dist = std::max(0.0f, euclidean_distance(a_tracks[i]->smooth_feat, b_tracks[j]->curr_feat));
                cost_matrix[i][j] = dist;

            } else if (distance_metric == "cosine") {
                dist = std::max(0.0f, cosine_distance(a_tracks[i]->smooth_feat, b_tracks[j]->curr_feat));
                cost_matrix[i][j] = dist;
            }

            // std::cout << "After distance calcs" << std::endl;
            // std::cout.flush();
            if (cost_matrix[i][j] > max_embedding_distance) {
                mask[i][j] = false;
            } else {
                mask[i][j] = true;
            }

        }        
    }

    return {cost_matrix, mask};
}

std::vector<std::vector<float>>
ImprAssoc_track::ImprAssocTracker::fuseDistances(std::vector<std::vector<float>> &iou_costs,
                                                const std::vector<std::vector<float>> &embed_costs,
                                                const std::vector<std::vector<bool>> &iou_mask,
                                                const float &iou_weight) const
{
    if (embed_costs.size() == 0) {
        for (int i=0; i<iou_costs.size(); i++) {
            for (int j=0; j<iou_costs[i].size(); j++) {
                if (iou_mask[i][j]) {
                    iou_costs[i][j] = 1.0F;
                }
            }
        }
        return iou_costs;
    }

    std::vector<std::vector<float>> fused_cost_matrix;
    fused_cost_matrix.resize(iou_costs.size());

    for (int i=0; i<iou_costs.size(); i++) {
        fused_cost_matrix[i].resize(iou_costs[i].size());
        for (int j=0; j<iou_costs[i].size(); j++) {
            if (iou_mask[i][j]) {
                fused_cost_matrix[i][j] = 1.0F;
            } else {
                fused_cost_matrix[i][j] = iou_weight * iou_costs[i][j] + (1-iou_weight) * embed_costs[i][j];
            }
        }
    }
    return fused_cost_matrix;
}

double ImprAssoc_track::ImprAssocTracker::execLapjv(const std::vector<std::vector<float>> &cost,
                                          std::vector<int> &rowsol,
                                          std::vector<int> &colsol,
                                          bool extend_cost,
                                          float cost_limit,
                                          bool return_cost) const
{
    std::vector<std::vector<float> > cost_c;
    cost_c.assign(cost.begin(), cost.end());

    std::vector<std::vector<float> > cost_c_extended;

    int n_rows = cost.size();
    int n_cols = cost[0].size();
    rowsol.resize(n_rows);
    colsol.resize(n_cols);

    int n = 0;
    if (n_rows == n_cols)
    {
        n = n_rows;
    }
    else
    {
        if (!extend_cost)
        {
            throw std::runtime_error("The `extend_cost` variable should set True");
        }
    }

    if (extend_cost || cost_limit < std::numeric_limits<float>::max())
    {
        n = n_rows + n_cols;
        cost_c_extended.resize(n);
        for (size_t i = 0; i < cost_c_extended.size(); i++)
            cost_c_extended[i].resize(n);

        if (cost_limit < std::numeric_limits<float>::max())
        {
            for (size_t i = 0; i < cost_c_extended.size(); i++)
            {
                for (size_t j = 0; j < cost_c_extended[i].size(); j++)
                {
                    cost_c_extended[i][j] = cost_limit / 2.0;
                }
            }
        }
        else
        {
            float cost_max = -1;
            for (size_t i = 0; i < cost_c.size(); i++)
            {
                for (size_t j = 0; j < cost_c[i].size(); j++)
                {
                    if (cost_c[i][j] > cost_max)
                        cost_max = cost_c[i][j];
                }
            }
            for (size_t i = 0; i < cost_c_extended.size(); i++)
            {
                for (size_t j = 0; j < cost_c_extended[i].size(); j++)
                {
                    cost_c_extended[i][j] = cost_max + 1;
                }
            }
        }

        for (size_t i = n_rows; i < cost_c_extended.size(); i++)
        {
            for (size_t j = n_cols; j < cost_c_extended[i].size(); j++)
            {
                cost_c_extended[i][j] = 0;
            }
        }
        for (int i = 0; i < n_rows; i++)
        {
            for (int j = 0; j < n_cols; j++)
            {
                cost_c_extended[i][j] = cost_c[i][j];
            }
        }

        cost_c.clear();
        cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
    }

    double **cost_ptr;
    cost_ptr = new double *[sizeof(double *) * n];
    for (int i = 0; i < n; i++)
        cost_ptr[i] = new double[sizeof(double) * n];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cost_ptr[i][j] = cost_c[i][j];
        }
    }

    int* x_c = new int[sizeof(int) * n];
    int *y_c = new int[sizeof(int) * n];

    int ret = lapjv_internal(n, cost_ptr, x_c, y_c);
    if (ret != 0)
    {
        throw std::runtime_error("The result of lapjv_internal() is invalid.");
    }

    double opt = 0.0;

    if (n != n_rows)
    {
        for (int i = 0; i < n; i++)
        {
            if (x_c[i] >= n_cols)
                x_c[i] = -1;
            if (y_c[i] >= n_rows)
                y_c[i] = -1;
        }
        for (int i = 0; i < n_rows; i++)
        {
            rowsol[i] = x_c[i];
        }
        for (int i = 0; i < n_cols; i++)
        {
            colsol[i] = y_c[i];
        }

        if (return_cost)
        {
            for (size_t i = 0; i < rowsol.size(); i++)
            {
                if (rowsol[i] != -1)
                {
                    opt += cost_ptr[i][rowsol[i]];
                }
            }
        }
    }
    else if (return_cost)
    {
        for (size_t i = 0; i < rowsol.size(); i++)
        {
            opt += cost_ptr[i][rowsol[i]];
        }
    }

    for (int i = 0; i < n; i++)
    {
        delete[]cost_ptr[i];
    }
    delete[]cost_ptr;
    delete[]x_c;
    delete[]y_c;

    return opt;
}
