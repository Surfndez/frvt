#ifndef OPENVINOINFERENCE_H_
#define OPENVINOINFERENCE_H_

#include <memory>
#include <opencv2/core.hpp>

#include "InferenceEngine.h"
#include "inference_engine.hpp"

using namespace InferenceEngine;

namespace FRVT_11 {
    class OpenVinoInference : public IInferenceEngine {
public:
    OpenVinoInference(const std::string &modelPath);

    void Init();

    std::shared_ptr<InferenceEngine::Blob> Infer(const cv::Mat& image);

private:
    std::string mModelPath;
    InferRequest m_infer_request;
    CNNNetwork m_network;
    CNNNetReader m_network_reader;
    std::string mInputName;
};
}

#endif /* OPENVINOINFERENCE_H_ */
