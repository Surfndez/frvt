#include "SsdFaceDetector.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "../algo/TimeMeasurement.h"

using namespace FRVT_11;

std::string MODEL_NAME = "/fd_r17_288-200000"; // ssd_resnet17_v1_288x288-200000
int SSD_INPUT_SIZE = 288;

SsdFaceDetector::SsdFaceDetector(const std::string &configDir)
{
    std::string modelPath = configDir + MODEL_NAME;

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
    auto t = TimeMeasurement();

    // Prepare image

    cv::Mat image(imageData.height, imageData.width, CV_8UC3, imageData.data.get());
    //cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

    //std::cout << "\tFace detection -> Create image "; t.Test();

    cv::resize(image, image, cv::Size(SSD_INPUT_SIZE, SSD_INPUT_SIZE), 0, 0, cv::INTER_LINEAR);

    //std::cout << "\tFace detection -> Resize image "; t.Test();

    float ratioH = imageData.height / float(image.rows);
    float ratioW = imageData.width / float(image.cols);

    // Perform inference

    auto output = mTensorFlowInference->Infer(image);

    //std::cout << "\tFace detection -> Inference "; t.Test();

    // Process output

    float* num_detections = static_cast<float*>(TF_TensorData(output[0].get()));
    float* scores = static_cast<float*>(TF_TensorData(output[1].get()));
    float* boxes = static_cast<float*>(TF_TensorData(output[2].get()));

    Rect rect(  int(boxes[1] * image.cols * ratioW),
                int(boxes[0] * image.rows * ratioH),
                int(boxes[3] * image.cols * ratioW),
                int(boxes[2] * image.rows * ratioH),
                scores[0]);

    //std::cout << "top score:         " << rect.score << std::endl;
    //std::cout << "face bounding box: [" <<  rect.x1 << "," << rect.y1 << "," << rect.x2 << "," << rect.y2 << "]" << std::endl;

    //std::cout << "\tFace detection -> Done "; t.Test();

    return {rect};
}
