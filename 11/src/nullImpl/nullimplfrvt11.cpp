/*
 * This software was developed at the National Institute of Standards and
 * Technology (NIST) by employees of the Federal Government in the course
 * of their official duties. Pursuant to title 17 Section 105 of the
 * United States Code, this software is not subject to copyright protection
 * and is in the public domain. NIST assumes no responsibility  whatsoever for
 * its use by other parties, and makes no guarantees, expressed or implied,
 * about its quality, reliability, or any other characteristic.
 */

#include <cstring>
#include <cstdlib>

#include <opencv2/core.hpp>

#include "nullimplfrvt11.h"
#include "../algo/SfdFaceDetector.h"
#include "../algo/FanLandmarksDetector.h"
#include "../algo/SphereFaceRecognizer.h"

using namespace std;
using namespace FRVT;
using namespace FRVT_11;

NullImplFRVT11::NullImplFRVT11() {}

NullImplFRVT11::~NullImplFRVT11() {}

ReturnStatus
NullImplFRVT11::initialize(const std::string &configDir)
{
    mFaceDetector = std::make_shared<SfdFaceDetector>(configDir);
    mLandmarksDetector = std::make_shared<FanLandmarksDetector>(configDir);
    mFaceRecognizer = std::make_shared<SphereFaceRecognizer>(configDir);

    return ReturnStatus(ReturnCode::Success);
}

ReturnStatus
NullImplFRVT11::createTemplate(
        const Multiface &faces,
        TemplateRole role,
        std::vector<uint8_t> &templ,
        std::vector<EyePair> &eyeCoordinates)
{
    std::vector<std::vector<float>> templates;

    for(const Image &image: faces) {        
        int channels = int(image.depth / 8);
        ImageData imageData(image.data, image.width, image.height, channels);

        std::vector<Rect> rects = mFaceDetector->Detect(imageData);

        for (const Rect &rect : rects) {
            std::vector<int> landmarks = mLandmarksDetector->Detect(imageData, rect);
            
            eyeCoordinates.push_back(EyePair(true, true, landmarks[0], landmarks[1], landmarks[2], landmarks[3]));

            std::vector<float> features = mFaceRecognizer->Infer(imageData, landmarks);

            templates.push_back(features);

            break; // should be only one face in image
        }        
    }

    // average pool on features
    cv::Mat output_features = cv::Mat::zeros(512, 1, CV_32F);
    for (const std::vector<float>& f : templates) {
        for (int i = 0; i < f.size(); ++i) {
            output_features.at<float>(i, 0) += f[i];
        }
    }

    output_features /= templates.size();
    output_features /= cv::norm(output_features);

    /* Note: example code, potentially not portable across machines. */
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>((const uint8_t*)output_features.data);
    int dataSize = sizeof(float) * output_features.rows;
    templ.resize(dataSize);
    memcpy(templ.data(), bytes, dataSize);

    return ReturnStatus(ReturnCode::Success);
}

ReturnStatus
NullImplFRVT11::matchTemplates(
        const std::vector<uint8_t> &verifTemplate,
        const std::vector<uint8_t> &enrollTemplate,
        double &similarity)
{
    std::cout << "Calculate similarity... ";

    cv::Mat f1(512, 1, CV_32F, (float *)verifTemplate.data());
    cv::Mat f2(512, 1, CV_32F, (float *)enrollTemplate.data());

    similarity = 300 * (3 - cv::norm(f1 - f2));

    std::cout << similarity << std::endl;

    return ReturnStatus(ReturnCode::Success);
}

std::shared_ptr<Interface>
Interface::getImplementation()
{
    return std::make_shared<NullImplFRVT11>();
}
