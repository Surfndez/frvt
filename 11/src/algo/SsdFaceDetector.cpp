#include "SsdFaceDetector.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace FRVT_11;

float DETECTION_SCORE = 0.3;

SsdFaceDetector::SsdFaceDetector(const std::string& configDir, const std::string& modelName, int inputSize) : mInputSize(inputSize)
{
    std::string modelPath = configDir + modelName;
    if (mOpenVino)
        // mModelInference = std::make_shared<OpenVinoInference>(OpenVinoInference(modelPath));
        return;
    else
        mTensorFlowInference = std::make_shared<TensorFlowInference>(TensorFlowInference(
            modelPath,
            {"image_tensor"},
            {"num_detections", "detection_scores", "detection_boxes", "detection_classes"})
        );
}

SsdFaceDetector::~SsdFaceDetector() {}

std::vector<Rect>
SsdFaceDetector::Detect(const cv::Mat& constImage) const
{
    float ratioH = float(constImage.rows);
    float ratioW = float(constImage.cols);

    if (mOpenVino)
    {
        // cv::Mat image;
        // constImage.convertTo(image, CV_32FC3);

        // cv::resize(image, image, cv::Size(mInputSize, mInputSize), 0, 0, cv::INTER_LINEAR);

        // auto output = mModelInference->Infer(image);

        // float *detection = output->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();
    
        // if (detection[2] > DETECTION_SCORE) {
        //     Rect rect(  int(detection[3] * ratioW),
        //                 int(detection[4] * ratioH),
        //                 int(detection[5] * ratioW),
        //                 int(detection[6] * ratioH),
        //                 detection[2]);
        //     return {rect};
        // }
    }
    else
    {
        auto output = mTensorFlowInference->Infer(constImage);
        
        float* num_detections = static_cast<float*>(TF_TensorData(output[0].get()));
        float* scores = static_cast<float*>(TF_TensorData(output[1].get()));
        float* boxes = static_cast<float*>(TF_TensorData(output[2].get()));
        
        if (scores[0] > DETECTION_SCORE) {
            Rect rect(  int(boxes[1] * ratioW),
                        int(boxes[0] * ratioH),
                        int(boxes[3] * ratioW),
                        int(boxes[2] * ratioH),
                        scores[0]);
            return {rect};
        }
    }
    
    return {};
}
