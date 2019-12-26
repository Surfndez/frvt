#ifndef SPHEREFACERECOGNIZER_H_
#define SPHEREFACERECOGNIZER_H_

#include <string>

#include "FaceRecognizer.h"
#include "OpenVinoInference.h"

namespace FRVT_11 {
    class SphereFaceRecognizer : public FaceRecognizer  {
public:
    SphereFaceRecognizer(const std::string &configDir);
    ~SphereFaceRecognizer() override;

    virtual std::vector<float> Infer(const cv::Mat& image) const override;

private:
    std::shared_ptr<OpenVinoInference> mModelInference;
};
}

#endif /* SPHEREFACERECOGNIZER_H_ */
