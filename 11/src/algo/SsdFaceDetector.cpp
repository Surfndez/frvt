#include "SsdFaceDetector.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace FRVT_11;

SsdFaceDetector::SsdFaceDetector(const std::string& configDir, const std::string& modelName, int inputSize) : mInputSize(inputSize)
{
    std::string modelPath = configDir + modelName;

    mModelInference = std::make_shared<OpenVinoInference>(OpenVinoInference(modelPath));
}

SsdFaceDetector::~SsdFaceDetector() {}

std::vector<Rect>
SsdFaceDetector::Detect(const ImageData &imageData) const
{
    cv::Mat image(imageData.height, imageData.width, CV_8UC3, imageData.data.get());
    return this->Detect(image);
}

std::vector<Rect>
SsdFaceDetector::Detect(const cv::Mat& constImage) const
{
    cv::Mat image;
    constImage.convertTo(image, CV_32FC3);

    float ratioH = float(image.rows);
    float ratioW = float(image.cols);

    cv::resize(image, image, cv::Size(mInputSize, mInputSize), 0, 0, cv::INTER_LINEAR);

    // Perform inference

    auto output = mModelInference->Infer(image);

    // Process output

    float *detection = output->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();
    
    if (detection[2] > 0.3) {
        Rect rect(  int(detection[3] * ratioW),
                    int(detection[4] * ratioH),
                    int(detection[5] * ratioW),
                    int(detection[6] * ratioH),
                    detection[2]);
        return {rect};
    }
    else {
        return {};
    }
}
