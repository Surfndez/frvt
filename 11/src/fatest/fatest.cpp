#include <iostream>
#include <stdexcept>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <iomanip>
#include <numeric>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "frvt11.h"
#include "util.h"

#include "../algo/TimeMeasurement.h"

using namespace FRVT;
using namespace FRVT_11;

class ProgressBarPrinter {
public:
    ProgressBarPrinter(int total_items) : init_time(true), total_items(total_items) {}
    
    void Print(int progress)
    {
        if (progress == 0)
        {
            std::cout << "Progress: 0% | 0/" << total_items << "\r" << std::flush;
        }
        else
        {
            if (init_time) // this is here so first inference and session creations are not calculated
            {
                start_time = std::chrono::high_resolution_clock::now();
                init_time = false;
            }

            int items_finished = int(progress / 3);
            int percentage_finished = int(items_finished / float(total_items) * 100);

            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start_time;
            double passed_time = elapsed.count();
            
            double time_per_item = passed_time / items_finished;
            times_per_item.push_back(time_per_item);
            if (times_per_item.size() > 20) times_per_item.erase(times_per_item.begin());
            time_per_item = std::accumulate(times_per_item.begin(), times_per_item.end(), 0.0) / times_per_item.size();
            
            double time_remaining = time_per_item * (total_items - items_finished);
            int minutes_remaining = int(time_remaining / 60);
            int seconds_remaining = int(time_remaining) % 60;

            std::cout
                << "Progress: "
                << percentage_finished << "% | " << items_finished << "/" << total_items
                << " | Remaining time: " << minutes_remaining << ":" << (seconds_remaining < 10 ? "0" : "") << seconds_remaining
                << " | Time per item: " << time_per_item
                << "\r" << std::flush;
        }
    }

private:
    using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;
    Time start_time;
    bool init_time;
    int total_items;
    std::vector<double> times_per_item;
};

void
InitializeImplementation(std::shared_ptr<Interface>& implPtr)
{
    implPtr->initialize("/home/administrator/nist/frvt/11/config");
}

Image
CvImageToImageData(const cv::Mat& image)
{
    std::shared_ptr<uint8_t> imageArray(new uint8_t[3 * image.rows * image.cols]);
    std::memcpy(imageArray.get(), image.data, 3 * image.rows * image.cols);

    Image imageData((uint16_t)image.cols, (uint16_t)image.rows, (uint8_t)24, imageArray, Image::Label::Unknown);
    
    return imageData;
}

void
SanityCheck()
{
    std::cout << "Strating sanity check..." << std::endl;

    cv::Mat image = cv::imread("/home/administrator/nist/frvt/common/images/S486-02-t10_01.ppm");
    if(!image.data) throw std::runtime_error("Could not open or find the image");

    std::cout << "Read image successfully" << std::endl;

    auto implPtr = Interface::getImplementation();

    std::cout << "Created fvrt interface" << std::endl;

    InitializeImplementation(implPtr);

    std::cout << "Initialized implementation" << std::endl;

    std::shared_ptr<uint8_t> imageArray(new uint8_t[3 * image.rows * image.cols]);
    std::memcpy(imageArray.get(), image.data, 3 * image.rows * image.cols);

    Image imageData = CvImageToImageData(image);
    FRVT::Multiface faces = {imageData};

    std::vector<uint8_t> templ;
    std::vector<EyePair> eyeCoordinates;

    implPtr->createTemplate(faces, TemplateRole::Enrollment_11, templ, eyeCoordinates);

    std::cout << "Sanity test done!\n" << std::endl;
}

double
CalculateTPR(double fpr_divider, std::vector<double>& diff_scores, std::vector<double>& same_scores)
{
    std::sort(diff_scores.begin(), diff_scores.end());
    std::reverse(diff_scores.begin(), diff_scores.end());
    int borderIndex = int(diff_scores.size() / fpr_divider);
    double borderScore = diff_scores[borderIndex];

    std::cout << "Border score: " << borderScore << " (at index " << borderIndex << ")" << std::endl;

    double tp = 0;
    for (int i = 0; i < same_scores.size(); ++i)
    {
        if (same_scores[i] > borderScore) ++tp;
    }

    return tp / same_scores.size();
}

void
RunVggTest()
{
    // Read test list
    std::ifstream is("/home/administrator/face_data/benchmarks/vgg_test_pairs.txt");
    std::istream_iterator<std::string> start(is), end;
    std::vector<std::string> testList(start, end);

    // Create FRVT implementation
    auto implPtr = Interface::getImplementation();
    InitializeImplementation(implPtr);

    // Loop over pairs

    std::vector<double> same_scores;
    std::vector<double> diff_scores;

    int total_items = int((testList.size() - 2) / 3);

    ProgressBarPrinter progress_bar(total_items);

    for (int i = 2, progress = 0; i < testList.size(); i = i + 3, progress = progress + 3)
    {
        progress_bar.Print(progress);

        auto path1 = "/home/administrator/face_data/benchmarks/original/" + testList[i];
        auto path2 = "/home/administrator/face_data/benchmarks/original/" + testList[i + 1];
        auto isSame = testList[i + 2] == "1";

        //std::cout << path1 << std::endl;
        cv::Mat image1 = cv::imread(path1);
        //std::cout << path2 << std::endl;
        cv::Mat image2 = cv::imread(path2);
        if(!image1.data || !image2.data) throw std::runtime_error("Could not open or find the image");

        Image imageData1 = CvImageToImageData(image1);
        Image imageData2 = CvImageToImageData(image2);

        std::vector<uint8_t> features1;
        std::vector<uint8_t> features2;
        std::vector<EyePair> eyeCoordinates1;
        std::vector<EyePair> eyeCoordinates2;

        implPtr->createTemplate({imageData1}, TemplateRole::Enrollment_11, features1, eyeCoordinates1);
        implPtr->createTemplate({imageData2}, TemplateRole::Enrollment_11, features2, eyeCoordinates2);

        double score = 0;
        implPtr->matchTemplates(features1, features2, score);

        if (isSame) same_scores.push_back(score);
        else diff_scores.push_back(score);
    }

    // Output TPR

    std::cout << std::endl;
    std::cout << "TPR @ FPR 1:" << 10 << " = " << CalculateTPR(10, diff_scores, same_scores) << std::endl;
    std::cout << "TPR @ FPR 1:" << 100 << " = " << CalculateTPR(100, diff_scores, same_scores) << std::endl;
    std::cout << "TPR @ FPR 1:" << 1000 << " = " << CalculateTPR(1000, diff_scores, same_scores) << std::endl;
}

int
main(int argc, char* argv[])
{
    //SanityCheck();
    RunVggTest();
	return 0;
}
