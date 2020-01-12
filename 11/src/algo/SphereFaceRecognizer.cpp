#include "SphereFaceRecognizer.h"

#include <opencv2/imgproc.hpp>

using namespace FRVT_11;

cv::Mat
NormalizeImage(const cv::Mat& image)
{
    cv::Mat inferImage;

    // to BGR
    cv::cvtColor(image, inferImage, cv::COLOR_RGB2BGR);
    
    inferImage.convertTo(inferImage, CV_32FC3);
    inferImage /= 255;
    inferImage -= cv::Scalar(0.5, 0.5, 0.5);
    return inferImage;
}

SphereFaceRecognizer::SphereFaceRecognizer(const std::string &configDir)
{
    std::string sphereModelPath = configDir + "/fa_model"; // sphereface_drop-sphere_v0_108_dm100_se_arcface_listv12_obj3_sim_01-175000_features
    if (mOpenVino)
        // mModelInference = std::make_shared<OpenVinoInference>(OpenVinoInference(sphereModelPath));
        return;
    else
        mTensorFlowInference = std::make_shared<TensorFlowInference>(TensorFlowInference(sphereModelPath, {"input"}, {"output_features"}));
}

SphereFaceRecognizer::~SphereFaceRecognizer() {}

cv::Mat
SphereFaceRecognizer::Extract(const cv::Mat& image) const
{
    if (mOpenVino)
    {
        // auto output = mModelInference ->Infer(image);
        // const auto result = output->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
        // return cv::Mat(512, 1, CV_32F, result);
    }
    else
    {
        auto output = mTensorFlowInference->Infer(image);
        float* output_features = static_cast<float*>(TF_TensorData(output[0].get()));
        return cv::Mat(512, 1, CV_32F, output_features).clone();
    }
}

std::vector<float>
SphereFaceRecognizer::Infer(const cv::Mat& constImage) const
{
    cv::Mat image = NormalizeImage(constImage);

    // infer on original image
    // cv::Mat featuresMat_1 = Extract(image);
    auto output = mTensorFlowInference->Infer(image);
    float* output_features = static_cast<float*>(TF_TensorData(output[0].get()));
    cv::Mat featuresMat_1(512, 1, CV_32F, output_features);

    // infer on flipped image
    // cv::flip(image, image, 1);
    // cv::Mat featuresMat_2 = Extract(image);

    // Average features
    cv::Mat featuresMat = featuresMat_1; // (featuresMat_1 + featuresMat_2) / 2;

    // convert to vector (function should be changed to return cv::Mat)
    std::vector<float> features(512);
    for (int i = 0; i < 512; ++i) {
        features[i] = featuresMat.at<float>(i, 0);
    }

    return features;
}
