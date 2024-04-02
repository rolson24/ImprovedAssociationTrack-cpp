#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// #include "BoTSORT.h"
#include "ImprAssocTrack/ImprAssocTracker.h"
#include "ImprAssocTrack/DataType.h"
// #include "GlobalMotionCompensation.h"
#include "ImprAssocTrack/STrack.h"
// #include ""


#define TEST_GMC 0
#define GT_AS_PREDS 1
#define YOLOv8_PREDS 0


/**
 * @brief Read detections from MOTChallenge format file
 * 
 * @param tracks Tracks
 * @param output_file Output file where the tracks will be written
 */
void mot_format_writer(const std::vector<ImprAssoc_track::ImprAssocTracker::STrackPtr> &tracks,
                       const std::string &output_file)
{
    std::ofstream mot_file(output_file, std::ios::app);
    for (const ImprAssoc_track::ImprAssocTracker::STrackPtr &track: tracks)
    {
        ImprAssoc_track::Rect<float> bbox_tlwh = track->getRect();

        mot_file << track->getFrameId() << "," << track->getTrackId() << ","
                 << bbox_tlwh.x() << "," << bbox_tlwh.y() << "," << bbox_tlwh.width()
                 << "," << bbox_tlwh.height() << ",-1,-1,-1,0" << std::endl;
    }
    mot_file.close();
}


/**
 * @brief Read detections from YOLOv8 format file
 * 
 * @param detection_file Detection file
 * @param frame_width Image width (used to convert normalized bounding box to absolute coordinates)
 * @param frame_height Image height (used to convert normalized bounding box to absolute coordinates)
 * @return std::vector<Detection> Detections
 */
std::vector<Detection>
read_detections_from_file(const std::string &detection_file, int frame_width,
                          int frame_height)
{
    std::vector<Detection> detections;
    std::ifstream det_file(detection_file);
    std::string line;
    while (std::getline(det_file, line))
    {
        std::istringstream iss(line);
        std::vector<float> values(std::istream_iterator<float>{iss},
                                  std::istream_iterator<float>());

        Detection det;
        if (values[0] != 0)
            continue;
        det.class_id = static_cast<int>(values[0]);
        det.bbox_tlwh =
                cv::Rect_(values[1] - values[3] / 2, values[2] - values[4] / 2,
                          values[3], values[4]);
        // bounding box is normalized, so convert to absolute coordinates
        det.bbox_tlwh.x *= frame_width;
        det.bbox_tlwh.y *= frame_height;
        det.bbox_tlwh.width *= frame_width;
        det.bbox_tlwh.height *= frame_height;

        det.confidence = values[5];
        detections.push_back(det);
    }
    det_file.close();
    return detections;
}


std::vector<std::vector<Detection>>
read_mot_gt_from_file(const std::string &gt_filepath)
{
    // https://stackoverflow.com/questions/57678677/multiple-object-tracking-mot-benchmark-data-set-format-for-ground-truth-tracki
    // BB format is: frame_no,object_id,bb_left,bb_top,bb_width,bb_height,score,X,Y,Z

    std::vector<std::vector<Detection>> all_gt_for_curr_sequence;
    std::ifstream gt_file(gt_filepath);
    std::string line;
    while (std::getline(gt_file, line))
    {
        std::istringstream iss(line);
        std::vector<float> values;

        while (std::getline(iss, line, ','))
        {
            values.push_back(std::stof(line));
        }

        Detection det;
        int frame_id = static_cast<int>(values[0]);
        det.class_id =
                0;// class_id is not provided in MOTChallenge format, so set it to 0 to indicate person
        det.bbox_tlwh = cv::Rect_(values[2], values[3], values[4], values[5]);
        det.confidence = static_cast<float>(values[6]) == 0
                                 ? 1.0f
                                 : static_cast<float>(values[6]);

        while (all_gt_for_curr_sequence.size() < frame_id)
        {
            all_gt_for_curr_sequence.emplace_back();
        }
        all_gt_for_curr_sequence[frame_id - 1].push_back(det);
    }

    return all_gt_for_curr_sequence;
}

std::vector<ImprAssoc_track::Object> convert_Detections_to_Objects(std::vector<Detection> gt_per_frame)
{
    std::vector<ImprAssoc_track::Object> objects;

    for (int j=0; j<gt_per_frame.size(); j++) {
        auto detection = gt_per_frame[j];
        auto cv_bbox = detection.bbox_tlwh;
        ImprAssoc_track::Rect<float> bbox_tlwh;
        bbox_tlwh.x() = cv_bbox.x;
        bbox_tlwh.y() = cv_bbox.y;
        bbox_tlwh.height() = cv_bbox.height;
        bbox_tlwh.width() = cv_bbox.width;
        ImprAssoc_track::Object object(bbox_tlwh, detection.class_id, detection.confidence);
        objects.push_back(object);
    }
    
    return objects;
}

/**
 * @brief Plot tracks on the frame
 * 
 * @param frame Input frame
 * @param detections Detections
 * @param tracks Tracks
 */
void plot_tracks(cv::Mat &frame, std::vector<Detection> &detections,
                 std::vector<ImprAssoc_track::ImprAssocTracker::STrackPtr> &tracks)
{
    static std::map<int, cv::Scalar> track_colors;
    cv::Scalar detection_color = cv::Scalar(0, 0, 0);
    for (const auto &det: detections)
    {
        cv::rectangle(frame, det.bbox_tlwh, detection_color, 1);
    }

    for (const ImprAssoc_track::ImprAssocTracker::STrackPtr &track: tracks)
    {
        ImprAssoc_track::Rect<float> bbox_tlwh = track->getRect();
        cv::Scalar color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);

        if (track_colors.find(track->getTrackId()) == track_colors.end())
        {
            track_colors[track->getTrackId()] = color;
        }
        else
        {
            color = track_colors[track->getTrackId()];
        }

        cv::rectangle(frame,
                      cv::Rect(static_cast<int>(bbox_tlwh.x()),
                               static_cast<int>(bbox_tlwh.y()),
                               static_cast<int>(bbox_tlwh.width()),
                               static_cast<int>(bbox_tlwh.height())),
                      color, 2);
        cv::putText(frame, std::to_string(track->getTrackId()),
                    cv::Point(static_cast<int>(bbox_tlwh.x()),
                              static_cast<int>(bbox_tlwh.y())),
                    cv::FONT_HERSHEY_SIMPLEX, 0.75, color, 2);

        cv::rectangle(frame, cv::Rect(10, 10, 20, 20), detection_color, -1);
        cv::putText(frame, "Detection", cv::Point(40, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.75, detection_color, 2);
    }
}


/**
 * @brief Check if the source is a video file (mp4, avi, mkv, webm)
 * 
 * @param source 
 * @return true 
 * @return false 
 */
bool check_source(const std::string &source)
{
    std::string ext = source.substr(source.find_last_of(".") + 1);
    if (ext == "mp4" || ext == "avi" || ext == "mkv" || ext == "webm")
    {
        return true;
    }
    return false;
}


int main(int argc, char **argv)
{

    // Command line arguments check
    if (argc < 4)
    {
        std::cout << "Usage eg. 1: ./ImprAssoctrack_inference <source> "
                     "<dir_containing_per_frame_detections> "
                     "<dir_to_save_mot_format_output>"
                  << std::endl;
        std::cout << "Usage eg. 2: ./ImprAssoctrack_inference "
                     "<reid_config_path> "
                     "<reid_onnx_model_path> "
                     "<source> <dir_containing_per_frame_detections> "
                     "<dir_to_save_mot_format_output> <gt_file>"
                  << std::endl;
        return -1;
    }
    else if (4 < argc && argc < 6)
    {
        std::cout << "Usage eg. 1: ./ImprAssoctrack_inference <source> "
                     "<dir_containing_per_frame_detections> "
                     "<dir_to_save_mot_format_output>"
                  << std::endl;
        std::cout << "Usage eg. 2: ./ImprAssoctrack_inference "
                     "<reid_config_path> "
                     "<reid_onnx_model_path> "
                     "<source> <dir_containing_per_frame_detections> "
                     "<dir_to_save_mot_format_output> <gt_file>"
                  << std::endl;
        return -1;
    }


    std::string reid_config_path, reid_onnx_model_path,
        source, labels_dir, output_dir, gt_filepath;

    if (argc == 4)
    {
        source = argv[1];
        labels_dir = argv[2];
        output_dir = argv[3];
    }
    else
    {
        reid_config_path = argv[1];
        reid_onnx_model_path = argv[2];
        source = argv[3];
        labels_dir = argv[4];
        output_dir = argv[5];
    }


    // Setup output directories
    std::string output_dir_mot = output_dir + "/mot";
    std::string output_dir_img = output_dir + "/img";
    std::filesystem::create_directories(output_dir_mot);
    std::filesystem::create_directories(output_dir_img);


// // // Initialize GlobalMotionCompensation
// #if (TEST_GMC == 1)
//     /*
//         {"orb", GMC_Method::ORB},
//         {"ecc", GMC_Method::ECC},
//         {"sparseOptFlow", GMC_Method::SparseOptFlow},
//         {"optFlowModified", GMC_Method::OptFlowModified},
//         {"OpenCV_VideoStab", GMC_Method::OpenCV_VideoStab},
//     */
//     GlobalMotionCompensation gmc = GlobalMotionCompensation(
//             GlobalMotionCompensation::GMC_method_map["OpenCV_VideoStab"], 2.0);
//     cv::VideoCapture cap("/home/vipin/datasets/test_videos/"
//                          "0x100000A9_424_20220427_094317.mp4");
//     cv::Mat frame;

//     while (cap.read(frame))
//     {
//         HomographyMatrix H = gmc.apply(frame, {});
//         cv::Mat H_cv;
//         cv::eigen2cv(H, H_cv);
//         cv::Mat warped_frame;
//         cv::warpPerspective(frame, warped_frame, H_cv, frame.size());

//         cv::imshow("frame", frame);
//         cv::imshow("warped", warped_frame);

//         if (cv::waitKey(1) == 27)
//         {
//             break;
//         }
//     }
//     return 0;
// #endif


    cv::Mat frame;
    cv::VideoCapture cap;
    int frame_counter = 0;
    double tracker_time_sum = 0, tracker_time_total = 0;
    std::string output_file_txt = output_dir_mot + "/all.txt";
    std::vector<std::string> image_filepaths;
    bool is_video = check_source(source);
    int fps = 30;
    int track_buffer = 30;

    // Initialize Impr_Assoc tracker
    std::unique_ptr<ImprAssoc_track::ImprAssocTracker> tracker;

    if (argc == 4)
    {
        std::cout << "Using default configs. No ReID..."
                  << std::endl;
        tracker = std::make_unique<ImprAssoc_track::ImprAssocTracker>(fps, track_buffer, "", "");
    }
    else
    {
        tracker = std::make_unique<ImprAssoc_track::ImprAssocTracker>(fps, track_buffer, reid_config_path,
                                            reid_onnx_model_path);
    }

    if (is_video)
    {
        cap = cv::VideoCapture(source);
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    }
    else
    {
        // Read filenames in labels dir
        for (const auto &entry: std::filesystem::directory_iterator(source))
        {
            image_filepaths.push_back(entry.path());
        }
        std::sort(image_filepaths.begin(), image_filepaths.end());
    }


#if (YOLOv8_PREDS == 1)
    // Read detections and execute MultiObjectTracker
    while (cap.read(frame) || frame_counter < image_filepaths.size())
    {
        std::string filename;

        if (is_video)
        {
            std::ostringstream ss;
            ss << std::setw(6) << std::setfill('0') << frame_counter;
            filename = ss.str();
        }
        else
        {
            frame = cv::imread(image_filepaths[frame_counter]);
            filename = image_filepaths[frame_counter].substr(
                    image_filepaths[frame_counter].find_last_of('/') + 1);
            filename = filename.substr(0, filename.find_last_of('.'));
        }

        std::string detection_file = labels_dir + "/" + filename + ".txt";
        std::vector<Detection> detections = read_detections_from_file(
                detection_file, frame.cols, frame.rows);

        std::vector<ImprAssoc_track::Object> objects = convert_Detections_to_Objects(detections);
        std::string output_file_img = output_dir_img + "/" + filename + ".jpg";

        // Execute tracker
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<ImprAssoc_track::ImprAssocTracker::STrackPtr> tracks =
                tracker->update(objects, frame);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        tracker_time_sum += elapsed.count();

        // Outputs
        mot_format_writer(tracks, output_file_txt);

        plot_tracks(frame, detections, tracks);
        cv::imwrite(output_file_img, frame);

        frame_counter++;

        if (frame_counter % 100 == 0)
        {
            std::cout << "Processed " << frame_counter << " frames\t";
            std::cout << "Tracker FPS (last 100 frames): "
                      << 100 / tracker_time_sum << std::endl;
            tracker_time_total += tracker_time_sum;
            tracker_time_sum = 0;
        }
    }
#endif


#if (GT_AS_PREDS == 1)
    std::vector<std::vector<Detection>> gt_per_frame =
            read_mot_gt_from_file(labels_dir);
    
    // std::vector<std::vector<ImprAssoc_track::Object>> objects = convert_Detections_to_Objects(gt_per_frame);

    while (cap.read(frame) || frame_counter < image_filepaths.size())
    {
        std::string filename;

        if (is_video)
        {
            std::ostringstream ss;
            ss << std::setw(6) << std::setfill('0') << frame_counter;
            filename = ss.str();
        }
        else
        {
            frame = cv::imread(image_filepaths[frame_counter]);
            filename = image_filepaths[frame_counter].substr(
                    image_filepaths[frame_counter].find_last_of('/') + 1);
            filename = filename.substr(0, filename.find_last_of('.'));
        }

        std::string output_file_img = output_dir_img + "/" + filename + ".jpg";

        std::vector<ImprAssoc_track::Object> objects = convert_Detections_to_Objects(gt_per_frame[frame_counter]);

        // Execute tracker
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<ImprAssoc_track::ImprAssocTracker::STrackPtr> tracks =
                tracker->update(objects, frame);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        tracker_time_sum += elapsed.count();

        // Outputs
        mot_format_writer(tracks, output_file_txt);

        plot_tracks(frame, gt_per_frame[frame_counter], tracks);
        cv::imwrite(output_file_img, frame);

        frame_counter++;

        if (frame_counter % 100 == 0)
        {
            tracker_time_total += tracker_time_sum;
            tracker_time_sum = 0;
        }
    }
#endif

    std::cout << "Average tracker FPS: " << frame_counter / tracker_time_total
              << std::endl;
    std::cout << "Average processing time per frame (ms): "
              << (tracker_time_total / frame_counter) * 1000 << std::endl;
    std::cout.flush();
    cap.release();

    return 0;
}

