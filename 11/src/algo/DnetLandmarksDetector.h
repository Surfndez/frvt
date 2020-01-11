#ifndef DNETLANDMARKSDETECTOR_H_
#define DNETLANDMARKSDETECTOR_H_

#include "LandmarksDetector.h"
#include "OpenVinoInference.h"

namespace FRVT_11 {
    class DnetLandmarksDetector : public LandmarksDetector {
public:
    DnetLandmarksDetector(const std::string &configDir);
    ~DnetLandmarksDetector() override;

    virtual std::vector<int> Detect(const cv::Mat& image, const Rect &face) const override;

private:
    std::shared_ptr<OpenVinoInference> mModelInference;
};
}

#endif /* DNETLANDMARKSDETECTOR_H_ */
