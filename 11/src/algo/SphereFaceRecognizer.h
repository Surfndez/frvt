#ifndef SPHEREFACERECOGNIZER_H_
#define SPHEREFACERECOGNIZER_H_

#include <string>

#include "FaceRecognizer.h"

namespace InferenceEngine {
    class InferencePlugin;
    class CNNNetwork;
    class ExecutableNetwork;
}

namespace FRVT_11 {
    class SphereFaceRecognizer : public FaceRecognizer  {
public:
    SphereFaceRecognizer(const std::string &configDir);
    ~SphereFaceRecognizer() override;

    virtual std::vector<float> Infer(const ImageData& imageData, const std::vector<int>& landmarks) const override;

private:
    std::shared_ptr<InferenceEngine::InferencePlugin> mInferencePlugin;
    std::shared_ptr<InferenceEngine::CNNNetwork> mCNNNetwork;
    std::shared_ptr<InferenceEngine::ExecutableNetwork> mExecutableNetwork;
    std::string mInputName;
    std::string mOutputName;
};
}

#endif /* SPHEREFACERECOGNIZER_H_ */
