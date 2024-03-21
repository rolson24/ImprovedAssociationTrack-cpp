#include "ImprAssocTrack/ImprAssocTracker.h"

#include "gtest/gtest.h"

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"
#include "boost/foreach.hpp"
#include "boost/optional.hpp"

#include <opencv2/opencv.hpp>

#include <cstddef>
#include <filesystem>


namespace
{
    constexpr double EPS = 1e-2;

    namespace fs = std::filesystem;

    const std::string D_RESULTS_FILE = "detection_results.json";
    const std::string T_RESULTS_FILE = "tracking_results.json";
    // std::cout << "Current directory: " << fs::current_path() << std::endl;
    const std::string VIDEO_PATH = "images/palace.mp4";
    const std::string REID_CONFIG_PATH = "../../config/reid.ini";
    // const std::string REID_MODEL_PATH = "../../models/mobilenetv2_x1_4_msmt17_788c12bc0c8c_TRT8601_CUDA12020_1_TF32_FP16.engine";
    const std::string REID_MODEL_PATH = "../../models/mobilenetv2_x1_4_msmt17.onnx";

    // key: track_id, value: rect of tracking object
    using ImprAssocTrackerOut = std::map<size_t, ImprAssoc_track::Rect<float>>;

    template <typename T>
    T get_data(const boost::property_tree::ptree &pt, const std::string &key)
    {
        T ret;
        if (boost::optional<T> data = pt.get_optional<T>(key))
        {
            ret = data.get();
        }
        else
        {
            throw std::runtime_error("Could not read the data from ptree: [key: " + key + "]");
        }
        return ret;
    }

    std::map<size_t, std::vector<ImprAssoc_track::Object>> get_inputs_ref(const boost::property_tree::ptree &pt)
    {
        std::map<size_t, std::vector<ImprAssoc_track::Object>> inputs_ref;
        BOOST_FOREACH (const boost::property_tree::ptree::value_type &child, pt.get_child("results"))
        {
            const boost::property_tree::ptree &result = child.second;
            const auto frame_id = get_data<int>(result, "frame_id");
            const auto prob = get_data<float>(result, "prob");
            const auto x = get_data<float>(result, "x");
            const auto y = get_data<float>(result, "y");
            const auto width = get_data<float>(result, "width");
            const auto height = get_data<float>(result, "height");

            decltype(inputs_ref)::iterator itr = inputs_ref.find(frame_id);
            if (itr != inputs_ref.end())
            {
                itr->second.emplace_back(ImprAssoc_track::Rect(x, y, width, height), 0, prob);
            }
            else
            {
                std::vector<ImprAssoc_track::Object> v(1, {ImprAssoc_track::Rect(x, y, width, height), 0, prob});
                inputs_ref.emplace_hint(inputs_ref.end(), frame_id, v);
            }
        }
        return inputs_ref;
    }

    std::map<size_t, ImprAssocTrackerOut> get_outputs_ref(const boost::property_tree::ptree &pt)
    {
        std::map<size_t, ImprAssocTrackerOut> outputs_ref;
        BOOST_FOREACH (const boost::property_tree::ptree::value_type &child, pt.get_child("results"))
        {
            const boost::property_tree::ptree &result = child.second;
            const auto frame_id = get_data<int>(result, "frame_id");
            const auto track_id = get_data<int>(result, "track_id");
            const auto x = get_data<float>(result, "x");
            const auto y = get_data<float>(result, "y");
            const auto width = get_data<float>(result, "width");
            const auto height = get_data<float>(result, "height");

            decltype(outputs_ref)::iterator itr = outputs_ref.find(frame_id);
            if (itr != outputs_ref.end())
            {
                itr->second.emplace(track_id, ImprAssoc_track::Rect<float>(x, y, width, height));
            }
            else
            {
                ImprAssocTrackerOut v{
                    {track_id, ImprAssoc_track::Rect<float>(x, y, width, height)},
                };
                outputs_ref.emplace_hint(outputs_ref.end(), frame_id, v);
            }
        }
        return outputs_ref;
    }
}

TEST(ImprAssocTrack, ImprAssocTracker)
{
    boost::property_tree::ptree pt_d_results;
    boost::property_tree::read_json(D_RESULTS_FILE, pt_d_results);

    boost::property_tree::ptree pt_t_results;
    boost::property_tree::read_json(T_RESULTS_FILE, pt_t_results);

    try
    {
        // Get infomation of reference data
        const auto detection_results_name = get_data<std::string>(pt_d_results, "name");
        const auto tracking_results_name = get_data<std::string>(pt_t_results, "name");
        const auto fps = get_data<int>(pt_d_results, "fps");
        const auto track_buffer = get_data<int>(pt_d_results, "track_buffer");

        if (detection_results_name != tracking_results_name)
        {
            throw std::runtime_error("The name of the tests are different: [detection_results_name: " + detection_results_name + 
                                     ", tracking_results_name: " + tracking_results_name + "]");
        }

        // Get input reference data from D_RESULTS_FILE
        const auto inputs_ref = get_inputs_ref(pt_d_results);

        // Get output reference data from T_RESULTS_FILE
        auto outputs_ref = get_outputs_ref(pt_t_results);

        // Open the video reader:
        cv::VideoCapture cap(VIDEO_PATH);
        if(!cap.isOpened()){
            throw std::runtime_error("Error opening video stream or file: " + VIDEO_PATH);
        }

        // Test ImprAssocTracker::update()
        ImprAssoc_track::ImprAssocTracker tracker(fps, track_buffer, REID_CONFIG_PATH, REID_MODEL_PATH);
        for (const auto &[frame_id, objects] : inputs_ref)
        {
            // Read the next frame
            std::cout << "Get ready to get frame" << std::endl;
            std::cout.flush();
            cv::Mat frame;
            cap >> frame;

            std::cout << "Got frame" << std::endl;
            std::cout.flush();

            // If the frame is empty, break immediately
            if (frame.empty())
                break;
            std::cout << "Get ready to update" << std::endl;
            std::cout.flush();

            const auto outputs = tracker.update(objects, frame);

            // Verify between the reference data and the output of the ImprAssocTracker impl
            EXPECT_EQ(outputs.size(), outputs_ref[frame_id].size());
            for (const auto &outputs_per_frame : outputs)
            {
                const auto &rect = outputs_per_frame->getRect();
                const auto &track_id = outputs_per_frame->getTrackId();
                const auto &ref = outputs_ref[frame_id][track_id];
                EXPECT_NEAR(ref.x(), rect.x(), EPS);
                EXPECT_NEAR(ref.y(), rect.y(), EPS);
                EXPECT_NEAR(ref.width(), rect.width(), EPS);
                EXPECT_NEAR(ref.height(), rect.height(), EPS);
            }
        }
    }
    catch (const std::exception &e)
    {
        FAIL() << e.what();
    }
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return (RUN_ALL_TESTS());
}
