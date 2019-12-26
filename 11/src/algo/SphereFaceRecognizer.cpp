#include "SphereFaceRecognizer.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace FRVT_11;

cv::Mat
CropImage(const cv::Mat& image, const std::vector<int>& landmarks)
{
    // Find min and max values from landmarks

    std::vector<float> xPoints;
    std::vector<float> yPoints;
    for (int i = 0; i < 10; i += 2) {
        xPoints.push_back(landmarks[i]);
        yPoints.push_back(landmarks[i+1]);
    }

    float xMin = *std::min_element(xPoints.begin(), xPoints.end());
    float xMax = *std::max_element(xPoints.begin(), xPoints.end());
    float yMin = *std::min_element(yPoints.begin(), yPoints.end());
    float yMax = *std::max_element(yPoints.begin(), yPoints.end());

    // std::cout << "\tMin/Max landmarks: " << xMin << " " << xMax << " " << yMin << " " << yMax << std::endl;
     
    float w = (xMax - xMin);
    float h = (yMax - yMin);

    // Calculate bounding box
    
    float x1 = xMin - (w * 0.75);
    float x2 = xMax + (w * 0.75);
    float y1 = yMin - (h * 0.75);
    float y2 = yMax + (h * 0.75);

    // Keep original face ratio

    w = x2 - x1;
    h = y2 - y1;
    if (h > w)
    {
        float c = (x1 + x2) / 2;
        float half_w = h / 2;
        x1 = c - half_w;
        x2 = c + half_w;
    }
    else
    {
        float c = (y1 + y2) / 2;
        float half_h = w / 2;
        y1 = c - half_h;
        y2 = c + half_h;
    }
    w = x2 - x1;
    h = y2 - y1;

    // std::cout << "\tKeep ratio rect: " << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;

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
    x1 = std::max(0.f, x1);
    x2 = std::min(float(image.cols), x2);
    y1 = std::max(0.f, y1);
    y2 = std::min(float(image.rows), y2);


   //  std::cout << "\tCrop coords: " << image.rows << " " << image.cols << " -> " << int(x1) << " " << int(y1) << " " << int(x2) << " " << int(y2) << std::endl;

    // Crop

    cv::Mat cropped = image(cv::Range(int(y1), int(y2)), cv::Range(int(x1), int(x2)));

    return cropped;
}

cv::Mat
NormalizeImage(const cv::Mat& image, const std::vector<int>& landmarks)
{
    // crop
    cv::Mat cropped = CropImage(image, landmarks);

    // std::cout << "\tCropped size: " << cropped.rows << " " << cropped.cols << std::endl;

    // resize
    cv::resize(cropped, cropped, cv::Size(128, 128), 0, 0, cv::INTER_AREA);

    // to BGR
    cv::cvtColor(cropped, cropped, cv::COLOR_RGB2BGR);

    // double min_i, max_i;
    // cv::minMaxLoc(cropped, &min_i, &max_i);
    // std::cout << "\tmin-max " << min_i << "-" << max_i << std::endl;
    
    cropped.convertTo(cropped, CV_32FC3);
    cropped /= 255;
    cropped -= cv::Scalar(0.5, 0.5, 0.5);

    // double min, max;
    // cv::minMaxLoc(cropped, &min, &max);
   //  std::cout << "\tmin-max " << min << "-" << max << std::endl;

    // cv::Scalar mean, stddev;
    // cv::meanStdDev (cropped, mean, stddev);
   //  std::cout << "\tMean-Std: " << mean << " " << stddev << std::endl;

    return cropped;
}

SphereFaceRecognizer::SphereFaceRecognizer(const std::string &configDir)
{
    std::string sphereModelPath = configDir + "/sphereface_drop-sphere_v0_108_dm100_se_arcface_listv9_obj_04-100000_features";
    mModelInference = std::make_shared<OpenVinoInference>(OpenVinoInference(sphereModelPath));
}

SphereFaceRecognizer::~SphereFaceRecognizer() {}

std::vector<float>
SphereFaceRecognizer::Infer(const ImageData& imageData, const std::vector<int>& landmarks) const
{
    cv::Mat image(imageData.height, imageData.width, CV_8UC3, imageData.data.get());
    
    image = NormalizeImage(image, landmarks);

    // cv::Mat flat = image.reshape(1, image.cols*image.rows*image.channels());
    // std::vector<float> vec = image.isContinuous()? flat : flat.clone();
    // std::ofstream fout("/home/administrator/nist/debug/data.dat", std::ios::out | std::ios::binary);
    // fout.write((char*)&vec[0], vec.size() * 4);
    // fout.close();

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
