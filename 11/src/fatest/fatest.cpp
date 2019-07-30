#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <cstdlib>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "frvt11.h"
#include "util.h"
#include "TestUtils.h"
#include "ProgressBar.h"

using namespace FRVT;
using namespace FRVT_11;

class TestIterator {
public:
    TestIterator(const std::string& list_path)
    {
        testList = ReadTestList(list_path);
        gallery_size = std::stoi(testList[0]);
        pair_size = gallery_size * 2 + 1;
        total_items = int((testList.size() - 1) / pair_size);
        std::cout << "Found gallery size: " << gallery_size << std::endl;
    }

    template<typename Functor>
    void Loop(Functor functor) const
    {
        for (int i = 1, progress = 0; i < testList.size(); i = i + pair_size, progress = progress + pair_size)
        {
            std::vector<std::string> files1(testList.begin() + i, testList.begin() + i + gallery_size);
            std::vector<std::string> files2(testList.begin() + i + gallery_size, testList.begin() + i + gallery_size * 2);

            auto isSame = testList[i + gallery_size * 2] == "1";

            functor(files1, files2, isSame, progress);
        }
    }

    int gallery_size;
    int pair_size;
    int total_items;

private:
    std::vector<std::string> testList;
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

double
GetThreshold(double fpr_divider, std::vector<double>& diff_scores, int& borderIndex)
{
    std::sort(diff_scores.begin(), diff_scores.end());
    std::reverse(diff_scores.begin(), diff_scores.end());
    borderIndex = int(diff_scores.size() / fpr_divider);
    double borderScore = diff_scores[borderIndex];
    return borderScore;
}

double
CalculateTPR(double fpr_divider, std::vector<double>& diff_scores, std::vector<double>& same_scores)
{
    int borderIndex;
    double borderScore = GetThreshold(fpr_divider, diff_scores, borderIndex);

    std::cout << "Threshold: " << borderScore << " (at index " << borderIndex << ")" << std::endl;

    double tp = 0;
    for (int i = 0; i < same_scores.size(); ++i)
    {
        if (same_scores[i] > borderScore) ++tp;
    }

    return tp / same_scores.size();
}

void
CalculateLandmarksAccuracy(std::map<std::string, std::vector<int>>& landmarksList, std::map<std::string, std::vector<int>>& landmarksDetection)
{
    if (landmarksDetection.size() == 0) {
        std::cout << "No landmarks detected - not calculating accuracy" << std::endl;
        return;
    }

    std::cout << "Calculating landmarks detections over " << landmarksDetection.size() << " images... ";
    std::vector<double> diffs;
    for (const auto& kv : landmarksDetection) {
        auto file = kv.first;
        auto landmarksDetected = kv.second;
        auto landmarksGT = landmarksList[file];
        double diff = 
            std::sqrt(std::pow(landmarksDetected[0] - landmarksGT[0], 2) + std::pow(landmarksDetected[1] - landmarksGT[1], 2)) +
            std::sqrt(std::pow(landmarksDetected[2] - landmarksGT[2], 2) + std::pow(landmarksDetected[3] - landmarksGT[3], 2)) ; 
        diff /= 2.0;
        diffs.push_back(diff);
    }

    auto maxDiff = *std::max_element(diffs.begin(), diffs.end());

    //cv::Mat testMat = cv::Mat(diffs);
    cv::Scalar mean, stddev;
    cv::meanStdDev(diffs, mean, stddev);

    std::cout << "Landmarks error: " << mean[0] << " +- " << stddev[0] << " (max: " << maxDiff << ")" << std::endl;
}

void
OutputFailedPairs(const TestIterator& test_iterator, const std::vector<double>& scores, std::vector<double>& diff_scores)
{
    std::ofstream f;
    f.open("/home/administrator/nist/frvt/debug/false_positives.txt");

    int borderIndex;
    double threshold = GetThreshold(1000, diff_scores, borderIndex);

    f << "Threshold: " << threshold << std::endl;

    int pair = 0;
    test_iterator.Loop([&] (const std::vector<std::string>& files1, const std::vector<std::string>& files2, bool isSame, int progress)
    {
        if (isSame && scores[pair] <= threshold)
        {
            f << "Same score: " << scores[pair] << std::endl;
            for (const auto& p : files1) f << p << " "; f << std::endl;
            for (const auto& p : files2) f << p << " "; f << std::endl;
        }
        if (!isSame && scores[pair] >= threshold)
        {
            f << "Diff score: " << scores[pair] << std::endl;
            for (const auto& p : files1) f << p << " "; f << std::endl;
            for (const auto& p : files2) f << p << " "; f << std::endl;
        } 
        ++pair;
    });

    f.close();
}

std::vector<uint8_t>
GetTemplate(std::shared_ptr<Interface>& implPtr, std::vector<std::string> files, std::map<std::string, std::vector<int>>& landmarksDetection)
{
    std::vector<Image> images;
    for (const std::string& file : files) {
        auto path = "/home/administrator/face_data/benchmarks/original/" + file;
        cv::Mat image = cv::imread(path);
        if(!image.data) throw std::runtime_error("Could not open or find the image");
        Image imageData = CvImageToImageData(image);
        images.push_back(imageData);
    }
    
    std::vector<uint8_t> features;
    std::vector<EyePair> eyeCoordinates;

    implPtr->createTemplate(images, TemplateRole::Enrollment_11, features, eyeCoordinates);

    if (files.size() == eyeCoordinates.size()) {
        for (int i = 0; i < files.size(); ++i) {
            EyePair eyePair = eyeCoordinates[i];
            landmarksDetection[files[i]] = {int(eyePair.xleft), int(eyePair.yleft), int(eyePair.xright), int(eyePair.yright)};
        }
    }

    return features;
}

void
RunVggTest(const std::string& list_path, const std::string& landmarks_list_path)
{
    // Load test list

    TestIterator test_iterator(list_path);

    // Load landmarks data

    std::map<std::string, std::vector<int>> landmarksList = ReadLandmarksList(landmarks_list_path);
    std::map<std::string, std::vector<int>> landmarksDetection;
    std::cout << "Loaded landmarks for " << landmarksList.size() << " images" << std::endl;

    // Create FRVT implementation

    auto implPtr = Interface::getImplementation();
    InitializeImplementation(implPtr);

    // Loop over pairs

    std::vector<double> same_scores;
    std::vector<double> diff_scores;
    std::vector<double> all_scores;

    ProgressBarPrinter progress_bar(test_iterator.total_items, test_iterator.gallery_size * 2);

    test_iterator.Loop([&] (const std::vector<std::string>& files1, const std::vector<std::string>& files2, bool isSame, int progress)
    {
        if (progress == 0) progress_bar.Print(progress);

        std::vector<uint8_t> features1 = GetTemplate(implPtr, files1, landmarksDetection);
        std::vector<uint8_t> features2 = GetTemplate(implPtr, files2, landmarksDetection);

        double score = 0;
        implPtr->matchTemplates(features1, features2, score);

        if (isSame) same_scores.push_back(score);
        else diff_scores.push_back(score);
        all_scores.push_back(score);

        if (progress == 0) progress_bar.RestartTime();
        if (progress > 0) progress_bar.Print(progress);
    });

    // Output TPR

    std::cout << std::endl;
    std::cout << "TPR @ FPR 1:" << 10 << " = " << CalculateTPR(10, diff_scores, same_scores) << std::endl;
    std::cout << "TPR @ FPR 1:" << 100 << " = " << CalculateTPR(100, diff_scores, same_scores) << std::endl;
    std::cout << "TPR @ FPR 1:" << 1000 << " = " << CalculateTPR(1000, diff_scores, same_scores) << std::endl;

    // Landmarks accuracy

    CalculateLandmarksAccuracy(landmarksList, landmarksDetection);

    // Dump analysis

    OutputFailedPairs(test_iterator, all_scores, diff_scores);
}

int
main(int argc, char* argv[])
{
    if (argc == 1) {
        std::cout << "Need test list path" << std::endl;
        return 1;
    }
    std::string listPath = argv[1];

    std::cout << "List path: " << listPath << std::endl;

    std::string landmarksListPath = "/home/administrator/face_data/benchmarks/vgg_landmarks.txt";
    
    RunVggTest(listPath, landmarksListPath);

	return 0;
}
