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
#include <stdlib.h>
#include <stdexcept>

#include <opencv2/core.hpp>

#include "nullimplfrvt11.h"
#include "../algo/SsdFaceDetector.h"
#include "../algo/DnetLandmarksDetector.h"
#include "../algo/SphereFaceRecognizer.h"

#include "../algo/TimeMeasurement.h"

using namespace std;
using namespace FRVT;
using namespace FRVT_11;

NullImplFRVT11::NullImplFRVT11() {}

NullImplFRVT11::~NullImplFRVT11() {}

void
CvMatToTemplate(const cv::Mat& mat, std::vector<uint8_t> &templ)
{
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>((const uint8_t*)mat.data);
    int dataSize = sizeof(float) * mat.rows;
    templ.resize(dataSize);
    memcpy(templ.data(), bytes, dataSize);
}

cv::Mat
AveragePoolOnTemplates(const std::vector<std::vector<float>>& templates)
{
    cv::Mat output_features = cv::Mat::zeros(512, 1, CV_32F);
    for (const std::vector<float>& f : templates) {
        for (int i = 0; i < f.size(); ++i) {
            output_features.at<float>(i, 0) += f[i];
        }
    }

    output_features /= templates.size();
    output_features /= cv::norm(output_features);

    return output_features;
}

ReturnStatus
NullImplFRVT11::initialize(const std::string &configDir)
{
    putenv("TF_CPP_MIN_LOG_LEVEL=3"); // Disable TensorFlow logs
    putenv("OMP_NUM_THREADS=1"); // Disable MKL muilti-threading
    cv::setNumThreads(0); // Disable OpenCV use of TBB

    mFaceDetector = std::make_shared<SsdFaceDetector>(configDir);
    mLandmarksDetector = std::make_shared<DnetLandmarksDetector>(configDir);
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
    try {
        auto result = safeCreateTemplate(faces, role, templ, eyeCoordinates);
        if (eyeCoordinates.size() == 0) throw std::runtime_error("no faces found");
        return result;
    }
    catch (const std::exception& e) {
        //std::cout << e.what() << std::endl;

        // fill dummy values
        cv::Mat output_features = cv::Mat::zeros(512, 1, CV_32F);
        CvMatToTemplate(output_features, templ);

        eyeCoordinates.clear();
        for(const Image &image: faces) eyeCoordinates.push_back(EyePair(true, true, 0, 0, 1, 1));

        return ReturnStatus(ReturnCode::UnknownError);
    }
}

ReturnStatus
NullImplFRVT11::safeCreateTemplate(
        const Multiface &faces,
        TemplateRole role,
        std::vector<uint8_t> &templ,
        std::vector<EyePair> &eyeCoordinates)
{
    std::vector<std::vector<float>> templates;

    for(const Image &image: faces) {
        int channels = int(image.depth / 8);
        ImageData imageData(image.data, image.width, image.height, channels);

        try {
            //auto t1 = TimeMeasurement();
            std::vector<Rect> rects = mFaceDetector->Detect(imageData);
            //std::cout << "Face detection "; t1.Test();
            const Rect& rect = rects[0]; // should be only one face in image

            //auto t2 = TimeMeasurement();
            std::vector<int> landmarks = mLandmarksDetector->Detect(imageData, rect);
            //std::cout << "Landmarks detection "; t2.Test();
            if (landmarks.size() == 0) continue;
            eyeCoordinates.push_back(EyePair(true, true, landmarks[0], landmarks[1], landmarks[2], landmarks[3]));

            //auto t3 = TimeMeasurement();
            std::vector<float> features = mFaceRecognizer->Infer(imageData, landmarks);
            //std::cout << "Face recognition "; t3.Test();

            templates.push_back(std::vector<float>(features.begin(), features.begin()+512));
            //templates.push_back(std::vector<float>(features.begin()+512, features.end()));
        }
        catch (const std::exception& e) {
            // Nothing to do for exceptions... move on to the next face...
            //std::cout << e.what() << std::endl;
        }
    }

    cv::Mat output_features = AveragePoolOnTemplates(templates);

    CvMatToTemplate(output_features, templ);

    return ReturnStatus(ReturnCode::Success);
}

ReturnStatus
NullImplFRVT11::matchTemplates(
        const std::vector<uint8_t> &verifTemplate,
        const std::vector<uint8_t> &enrollTemplate,
        double &similarity)
{
    cv::Mat f1(512, 1, CV_32F, (float *)verifTemplate.data());
    cv::Mat f2(512, 1, CV_32F, (float *)enrollTemplate.data());

    similarity = 300 * (3 - cv::norm(f1 - f2));

    return ReturnStatus(ReturnCode::Success);
}

std::shared_ptr<Interface>
Interface::getImplementation()
{
    return std::make_shared<NullImplFRVT11>();
}
