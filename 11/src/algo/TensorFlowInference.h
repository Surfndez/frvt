#ifndef TENSORFLOWINFERENCE_H_
#define TENSORFLOWINFERENCE_H_

#include <string>
#include <opencv2/core.hpp>

#include "InferenceEngine.h"

namespace FRVT_11 {
    class TensorFlowInference : public IInferenceEngine {
public:
    TensorFlowInference(const std::string &modelPath);

    void Infer(const cv::Mat& image) const;
};
}

#endif /* TENSORFLOWINFERENCE_H_ */