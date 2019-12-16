#include "SsdFaceDetector.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace FRVT_11;

SsdFaceDetector::SsdFaceDetector(const std::string& configDir, const std::string& modelName, int inputSize) : mInputSize(inputSize)
{
    std::string modelPath = configDir + modelName;

    mModelInference = std::make_shared<OpenVinoInference>(OpenVinoInference(modelPath, {"image_tensor"},
        {"num_detections", "detection_scores", "detection_boxes", "detection_classes"}));
}

SsdFaceDetector::~SsdFaceDetector() {}

std::vector<Rect>
SsdFaceDetector::Detect(const ImageData &imageData) const
{
    // Prepare image

    cv::Mat image(imageData.height, imageData.width, CV_8UC3, imageData.data.get());
    image.convertTo(image, CV_32FC3);

    float ratioH = float(image.rows);
    float ratioW = float(image.cols);

    cv::resize(image, image, cv::Size(mInputSize, mInputSize), 0, 0, cv::INTER_LINEAR);

    // Perform inference

    auto output = mModelInference->Infer(image);

    // Process output
    
    std::cout << std::endl << std::endl;

    float *detection = output->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();
    
    if (detection[2] > 0.3) {
        Rect rect(  int(detection[3] * ratioW),
                    int(detection[4] * ratioH),
                    int(detection[5] * ratioW),
                    int(detection[6] * ratioH),
                    detection[2]);
        std::cout << "\tConfidence: " << rect.score << std::endl;
        std::cout << "\tRect: " << rect.x1 << " " << rect.y1 << " " << rect.x2 << " " << rect.y2 << std::endl;
        std::cout << std::endl;
        return {rect};
    }
    else {
        return {};
    }
}
