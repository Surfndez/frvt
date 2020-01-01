#ifndef SSDFACEDETECTOR_H_
#define SSDFACEDETECTOR_H_

#include "OpenVinoInference.h"
#include "FaceDetector.h"

namespace FRVT_11 {
    class SsdFaceDetector : public FaceDetector  {
public:
    SsdFaceDetector(const std::string &configDir, const std::string& modelName, int inputSize);
    ~SsdFaceDetector() override;

    virtual std::vector<Rect> Detect(const ImageData& image) const override;

    virtual std::vector<Rect> Detect(const cv::Mat& image) const override;

private:
    std::shared_ptr<OpenVinoInference> mModelInference;
    int mInputSize;
};
}

#endif /* SSDFACEDETECTOR_H_ */