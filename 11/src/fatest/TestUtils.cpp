#include "TestUtils.h"

#include <iostream>
#include <fstream>
#include <iterator>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

std::vector<std::string>
ReadTestList(const std::string& list_path)
{
    std::ifstream is(list_path);
    std::istream_iterator<std::string> start(is), end;
    std::vector<std::string> testList(start, end);
    return testList;
}

 std::map<std::string, std::vector<int>>
 ReadLandmarksList(const std::string& list_path)
 {
    std::ifstream is(list_path);
    std::istream_iterator<std::string> start(is), end;
    std::vector<std::string> landmarks_list(start, end);

    std::map<std::string, std::vector<int>> landmarks_map;

    for (int i = 0; i < landmarks_list.size(); i = i + 11)
    {
        auto key = landmarks_list[i];
        
        std::vector<int> landmarks;
        for (int j = 1; j < 11; ++j)
        {
            landmarks.push_back(std::stoi(landmarks_list[i + j]));
        }

        landmarks_map[key] = landmarks;
    }

    return landmarks_map;
 }

std::vector<std::string>
SplitString(const std::string& str)
{
    std::string buf;                 // Have a buffer string
    std::stringstream ss(str);       // Insert the string into a stream

    std::vector<std::string> tokens; // Create vector to hold our words

    while (ss >> buf)
        tokens.push_back(buf);

    return tokens;
}

cv::Mat
LoadImage(const std::string& path)
{
    cv::Mat image = cv::imread(path);
    if(!image.data) throw std::runtime_error("Could not open or find the image");
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // This simulates the images that we get in NIST which are in RGB format
    return image;
}

// Accuracy

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
