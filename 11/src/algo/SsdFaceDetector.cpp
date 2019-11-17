#include "SsdFaceDetector.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace FRVT_11;

SsdFaceDetector::SsdFaceDetector(const std::string& configDir, const std::string& modelName, int inputSize) : mInputSize(inputSize)
{
    std::string modelPath = configDir + modelName;

    mTensorFlowInference = std::make_shared<TensorFlowInference>(TensorFlowInference(
        modelPath,
        {"image_tensor"},
        {"num_detections", "detection_scores", "detection_boxes", "detection_classes"})
    );
}

SsdFaceDetector::~SsdFaceDetector() {}

std::vector<Rect>
SsdFaceDetector::Detect(const ImageData &imageData) const
{
    // Prepare image

    cv::Mat image(imageData.height, imageData.width, CV_8UC3, imageData.data.get());

    float ratioH = float(image.rows);
    float ratioW = float(image.cols);

    // Perform inference

    auto output = mTensorFlowInference->Infer(image);

    // Process output

    float* num_detections = static_cast<float*>(TF_TensorData(output[0].get()));
    float* scores = static_cast<float*>(TF_TensorData(output[1].get()));
    float* boxes = static_cast<float*>(TF_TensorData(output[2].get()));

    if (scores[0] > 0.3) {
        Rect rect(  int(boxes[1] * ratioW),
                    int(boxes[0] * ratioH),
                    int(boxes[3] * ratioW),
                    int(boxes[2] * ratioH),
                    scores[0]);
        return {rect};
    }
    else {
        return {};
    }
}
