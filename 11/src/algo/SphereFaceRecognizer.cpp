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
    // Find min and max values from landmarks

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

    // Calculate bounding box
    
    int x1 = xMin - int(w * 0.75);
    int x2 = xMax + int(w * 0.75);
    int y1 = yMin - int(h * 0.75);
    int y2 = yMax + int(h * 0.75);

    // Keep original face ratio

    w = x2 - x1;
    h = y2 - y1;
    if (h > w)
    {
        int c = int((x1 + x2) / 2);
        int half_w = int(h / 2);
        x1 = c - half_w;
        x2 = c + half_w;
    }
    else
    {
        int c = int((y1 + y2) / 2);
        int half_h = int(w / 2);
        y1 = c - half_h;
        y2 = c + half_h;
    }
    w = x2 - x1;
    h = y2 - y1;

    // Check if need to pad image
    // std::vector<int> xy = {-x1, -y1, x2 + 1 - w, y2 + 1 - h};
    // int out_of_border = *std::max_element(xy.begin(), xy.end());
    // if (out_of_border > 0)
    // {
    //     cv::copyMakeBorder(image, image, out_of_border, out_of_border, out_of_border, out_of_border, cv::BORDER_CONSTANT, cv::mean(image).val[0]);
    //     x1 += out_of_border;
    //     x2 += out_of_border;
    //     y1 += out_of_border;
    //     y2 += out_of_border;
    // }

    // TODO..... remove once padding is done
    x1 = std::max(0, x1);
    x2 = std::min(image.cols, x2);
    y1 = std::max(0, y1);
    y2 = std::min(image.rows, y2);

    // Crop

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
