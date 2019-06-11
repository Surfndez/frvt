#include "SphereFaceRecognizer.h"

#include <algorithm>
#include <iostream>
//#include <inference_engine.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

//#include <opencv2/highgui.hpp>

#include "../algo/TimeMeasurement.h"

using namespace FRVT_11;
//using namespace InferenceEngine;

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

    // resize
    cv::resize(gray, gray, cv::Size(128, 128), 0, 0, cv::INTER_LINEAR);

    //cv::imwrite("/home/administrator/nist/frvt/debug/fa_gray_128.png", gray);

    // normalized
    gray.convertTo(gray, CV_32FC1);
    gray /= 255;
    gray -= 0.5;

    return gray;
}

SphereFaceRecognizer::SphereFaceRecognizer(const std::string &configDir)
{
    std::string sphereModelPath = configDir + "/fa_108_31-400000.pb"; // sphereface-sphereface_108_cosineface_nist_bbox_31-400000
    mTensorFlowInference = std::make_shared<TensorFlowInference>(TensorFlowInference(sphereModelPath, {"input"}, {"output_features"}));
    //mModelInference = std::make_shared<OpenVinoInference>(sphereModelPath);
}

SphereFaceRecognizer::~SphereFaceRecognizer() {}

std::vector<float>
SphereFaceRecognizer::Infer(const ImageData& imageData, const std::vector<int>& landmarks) const
{
    auto t = TimeMeasurement();

    cv::Mat image(imageData.height, imageData.width, CV_8UC3, imageData.data.get());
    
    image = NormalizeImage(image, landmarks);

    //std::cout << "\tFace recognition -> Prepare input "; t.Test();

    //auto output = mModelInference->Infer(image);
    //const auto result = output->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
    auto output = mTensorFlowInference->Infer(image);
    float* output_features = static_cast<float*>(TF_TensorData(output[0].get()));

    //std::cout << "\tFace recognition -> Inference "; t.Test();

    //std::cout << "features[:5] = " << output_features[0] << "," << output_features[1] << "," << output_features[2] << "," << output_features[3] << "," << output_features[4] << std::endl;

    // normalize vector
    cv::Mat featuresMat(512, 1, CV_32F, output_features);
    //featuresMat /= cv::norm(featuresMat);

    //std::cout << "embeddings[:5] = " << featuresMat.at<float>(0, 0) << "," << featuresMat.at<float>(1, 0) << "," << featuresMat.at<float>(2, 0) << "," << featuresMat.at<float>(3, 0) << "," << featuresMat.at<float>(4, 0) << std::endl;

    // convert to vector (function should be changed to return cv::Mat)
    std::vector<float> features(512);
    for (int i = 0; i < 512; ++i) {
        features[i] = featuresMat.at<float>(i, 0);
    }

    //std::cout << "\tFace recognition -> Done "; t.Test();

    return features;
}
