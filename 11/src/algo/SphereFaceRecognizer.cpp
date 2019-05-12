#include "SphereFaceRecognizer.h"

#include <algorithm>
#include <inference_engine.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>

using namespace FRVT_11;
using namespace InferenceEngine;

cv::Mat
CropImage(const cv::Mat& image, const std::vector<int>& landmarks)
{
    std::vector<int> xPoints;
    std::vector<int> yPoints;
    for (int i = 0; i < 10; i += 2) {
        xPoints.push_back(landmarks[i]);
        yPoints.push_back(landmarks[i+1]);
    }

    int xMin = *std::min_element(xPoints.begin(), xPoints.end());
    int xMax = *std::max_element(xPoints.begin(), xPoints.end());
    int yMin = *std::min_element(yPoints.begin(), yPoints.end());
    int yMax = *std::max_element(yPoints.begin(), yPoints.end());
    
    int w = (xMax - xMin);
    int h = (yMax - yMin);
    
    int x1 = xMin - int(w * 0.75);
    int x2 = xMax + int(w * 0.75);
    int y1 = yMin - int(h * 0.75);
    int y2 = yMax + int(h * 0.75);

    x1 = std::max(0, x1);
    x2 = std::min(image.cols, x2);
    y1 = std::max(0, y1);
    y2 = std::min(image.rows, y2);

    cv::Mat cropped = image(cv::Range(y1, y2), cv::Range(x1, x2));

    //cv::imwrite("/home/administrator/nist/frvt/debug/fa_cropped.png", cropped);

    return cropped;
}

cv::Mat
NormalizeImage(const cv::Mat& image, const std::vector<int>& landmarks)
{
    // crop
    cv::Mat cropped = CropImage(image, landmarks);

    // To gray scale
    cv::Mat gray;
    cv::cvtColor(cropped, gray, cv::COLOR_RGB2GRAY);

    // normalized
    gray.convertTo(gray, CV_32FC1);
    gray /= 255;
    gray -= 0.5;

    // resize
    cv::resize(gray, gray, cv::Size(128, 128), 0, 0, cv::INTER_LINEAR);

    return gray;
}

SphereFaceRecognizer::SphereFaceRecognizer(const std::string &configDir)
{
    std::string sphereModelPath = configDir + "/sphereface_v3-sphereface_v3_28_dm120_cosineface_bbox_0-4075000_features";
    mModelInference = std::make_shared<OpenVinoInference>(sphereModelPath);
}

SphereFaceRecognizer::~SphereFaceRecognizer() {}

std::vector<float>
SphereFaceRecognizer::Infer(const ImageData& imageData, const std::vector<int>& landmarks) const
{
    std::cout << "Sphere inference... " << std::endl;

    cv::Mat image(imageData.height, imageData.width, CV_8UC3, imageData.data.get());
    
    image = NormalizeImage(image, landmarks);

    auto output = mModelInference->Infer(image);
    const auto result = output->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

    // normalize vector
    cv::Mat featuresMat(512, 1, CV_32F, result);
    featuresMat /= cv::norm(featuresMat);

    // convert to vector (function should be changed to return cv::Mat)
    std::vector<float> features(featuresMat.data, featuresMat.data + 512);

    std::cout << "Done!" << std::endl;

    return features;
}
