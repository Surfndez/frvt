#include "SphereFaceRecognizer.h"

#include <algorithm>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "../algo/TimeMeasurement.h"

using namespace FRVT_11;

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

    //std::cout << "orig h w: " << image.rows << " " << image.cols << std::endl;
    //std::cout << "crop box: " << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;

    cv::Mat cropped = image(cv::Range(y1, y2), cv::Range(x1, x2));

    return cropped;
}

cv::Mat
NormalizeImage(const cv::Mat& image, const std::vector<int>& landmarks)
{
    // crop
    cv::Mat cropped = CropImage(image, landmarks);

    //std::cout << "cropped size: " << cropped.rows << " " << cropped.cols << std::endl;

    // resize
    cv::resize(cropped, cropped, cv::Size(128, 128), 0, 0, cv::INTER_LINEAR);
    
    cropped.convertTo(cropped, CV_32FC3);
    cropped /= 255;
    cropped -= cv::Scalar(0.5, 0.5, 0.5);

    //double min, max;
    //cv::minMaxLoc(cropped, &min, &max);
    //std::cout << "min-max " << min << "-" << max << std::endl;

    return cropped;
}

SphereFaceRecognizer::SphereFaceRecognizer(const std::string &configDir)
{
    std::string sphereModelPath = configDir + "/fa_v0_108_01-1375000"; // sphereface-sphere_v0_108_dm100_se_arcface_sqrbox_rgb_01-1375000
    mTensorFlowInference = std::make_shared<TensorFlowInference>(TensorFlowInference(sphereModelPath, {"input"}, {"embeddings"}));
}

SphereFaceRecognizer::~SphereFaceRecognizer() {}

std::vector<float>
SphereFaceRecognizer::Infer(const ImageData& imageData, const std::vector<int>& landmarks) const
{
    cv::Mat image(imageData.height, imageData.width, CV_8UC3, imageData.data.get());

    //std::cout << "1: h w : " << image.rows << " " << image.cols << std::endl;
    
    image = NormalizeImage(image, landmarks);

    // infer on original image
    auto output = mTensorFlowInference->Infer(image);
    float* output_features = static_cast<float*>(TF_TensorData(output[0].get()));
    cv::Mat featuresMat_1(512, 1, CV_32F, output_features);

    // infer on flipped image
    // cv::flip(image, image, 1);
    // auto output_2 = mTensorFlowInference->Infer(image);
    // float* output_features_2 = static_cast<float*>(TF_TensorData(output_2[0].get()));
    // cv::Mat featuresMat_2(512, 1, CV_32F, output_features_2);

    // convert to vector (function should be changed to return cv::Mat)
    std::vector<float> features(512);
    for (int i = 0; i < 512; ++i) {
        features[i] = featuresMat_1.at<float>(i, 0);
        // features[i+512] = featuresMat_2.at<float>(i, 0);
    }

    //std::cout << features[0] << " " << features[1] << std::endl;

    return features;
}
