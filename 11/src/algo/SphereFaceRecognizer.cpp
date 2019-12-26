#include "SphereFaceRecognizer.h"

using namespace FRVT_11;

SphereFaceRecognizer::SphereFaceRecognizer(const std::string &configDir)
{
    std::string sphereModelPath = configDir + "/sphereface_drop-sphere_v0_108_dm100_se_arcface_listv9_obj_04-100000_features";
    mModelInference = std::make_shared<OpenVinoInference>(OpenVinoInference(sphereModelPath));
}

SphereFaceRecognizer::~SphereFaceRecognizer() {}

std::vector<float>
SphereFaceRecognizer::Infer(const cv::Mat& image) const
{
    // infer on original image
    auto output = mModelInference ->Infer(image);
    const auto result = output->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
    cv::Mat featuresMat_1(512, 1, CV_32F, result);

    // infer on flipped image
    cv::flip(image, image, 1);
    auto output_2 = mModelInference ->Infer(image);
    const auto result1 = output_2->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
    cv::Mat featuresMat_2(512, 1, CV_32F, result1);

    // Average and re-norm features
    cv::Mat featuresMat = (featuresMat_1 + featuresMat_2) / 2;
    featuresMat /= cv::norm(featuresMat);

    // convert to vector (function should be changed to return cv::Mat)
    std::vector<float> features(512);
    for (int i = 0; i < 512; ++i) {
        features[i] = featuresMat.at<float>(i, 0);
    }

    return features;
}
