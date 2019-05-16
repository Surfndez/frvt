#ifndef TENSORFLOWINFERENCE_H_
#define TENSORFLOWINFERENCE_H_

#include <string>
#include <memory>
#include <opencv2/core.hpp>

#include "InferenceEngine.h"

extern "C"
{
    #include "c_api.h"
}

namespace FRVT_11 {
    class TensorFlowInference : public IInferenceEngine {
public:
    TensorFlowInference(const std::string &modelPath, std::initializer_list<std::string> inputLayers, std::initializer_list<std::string> outputLayers);

    void Infer(const cv::Mat& image);

private:
    TF_Graph* graph;
    TF_Status* status;
    TF_SessionOptions* options;
    TF_Session* sess;
};
}

#endif /* TENSORFLOWINFERENCE_H_ */
