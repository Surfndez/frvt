#ifndef SSDFACEDETECTOR_H_
#define SSDFACEDETECTOR_H_

#include "TensorFlowInference.h"
#include "FaceDetector.h"

namespace FRVT_11 {
    class SsdFaceDetector : public FaceDetector  {
public:
    SsdFaceDetector(const std::string &configDir);
    ~SsdFaceDetector() override;

    virtual std::vector<Rect> Detect(const ImageData &image) const override;

private:
    std::shared_ptr<TensorFlowInference> mTensorFlowInference;
};
}

#endif /* SSDFACEDETECTOR_H_ */
