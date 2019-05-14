#ifndef OPENVINOINFERENCE_H_
#define OPENVINOINFERENCE_H_

#include <memory>
#include <opencv2/core.hpp>

#include "InferenceEngine.h"

namespace InferenceEngine {
    class InferencePlugin;
    class CNNNetwork;
    class ExecutableNetwork;
    class InferRequest;
    class Blob;
}

namespace FRVT_11 {
    class OpenVinoInference : public IInferenceEngine {
public:
    OpenVinoInference(const std::string &modelPath);

    std::shared_ptr<InferenceEngine::Blob> Infer(const cv::Mat& image) const;


private:
    std::shared_ptr<InferenceEngine::InferencePlugin> mInferencePlugin;
    std::shared_ptr<InferenceEngine::CNNNetwork> mCNNNetwork;
    std::shared_ptr<InferenceEngine::ExecutableNetwork> mExecutableNetwork;
    std::shared_ptr<InferenceEngine::InferRequest> mInferRequest;
    std::string mInputName;
    std::string mOutputName;
};
}

#endif /* OPENVINOINFERENCE_H_ */
