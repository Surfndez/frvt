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
#include <fstream>

#include <opencv2/imgproc.hpp>

#include "nullimplfrvt11.h"
#include "../algo/FaceDetectionEnsemble.h"
#include "../algo/DnetLandmarksDetector.h"
#include "../algo/SphereFaceRecognizer.h"

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

    mFaceDetector = std::make_shared<FaceDetectionEnsemble>(configDir);
    mLandmarksDetector = std::make_shared<DnetLandmarksDetector>(configDir);
    mImageNormalizer = std::make_shared<ImageNormalizer>();
    mFaceClassifier = std::make_shared<FaceClassifier>(configDir);
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
        return safeCreateTemplate(faces, role, templ, eyeCoordinates);
    }
    catch (const std::exception& e) {
        //std::cout << e.what() << std::endl;
        templ.clear();
        eyeCoordinates.clear();
        return ReturnStatus(ReturnCode::UnknownError);
    }
}

void
DebugPrint(const Rect& rect, const std::vector<int>& landmarks, std::vector<float> features)
{
    std::ofstream dataFile("flow_data.txt", std::ios::out | std::ios::app);

    dataFile << "Rectangle: " << rect.x1 << " " << rect.y1 << " " << rect.x2 << " " << rect.y2 << std::endl;

    dataFile << "Landmarks:";
    for (int i = 0; i < 10; i++) dataFile << " " << landmarks[i];
    dataFile << std::endl;

    cv::Mat f1(512, 1, CV_32F, features.data());
    float norm = cv::norm(f1);
    f1 /= norm;
    dataFile << "Features:";
    for (int i=0; i < 512; i++) dataFile << " " << ((float *)f1.data)[i];
    dataFile << std::endl;

    dataFile << "Norm: " << norm << std::endl;
}

float
ResizeImage(cv::Mat& image)
{
    float maxImageSize = 512.;
    if (image.cols > maxImageSize && image.rows > maxImageSize)
    {
        float ratio = std::max(maxImageSize/image.cols, maxImageSize/image.rows);
        cv::resize(image, image, cv::Size(), ratio, ratio, cv::INTER_LINEAR);
        return ratio;
    }
    return 1.;
}

ReturnStatus
NullImplFRVT11::safeCreateTemplate(
        const Multiface &faces,
        TemplateRole role,
        std::vector<uint8_t> &templ,
        std::vector<EyePair> &eyeCoordinates)
{
    std::vector<std::vector<float>> templates;

    for(const Image &face: faces) {
        int channels = int(face.depth / 8);
        cv::Mat image(face.height, face.width, CV_8UC3, face.data.get());
        float ratio = ResizeImage(image);

        try {
            std::vector<Rect> rects = mFaceDetector->Detect(image);
            if (rects.size() == 0) continue;
            const Rect& rect = rects[0]; // should be only one face in image

            std::vector<int> landmarks = mLandmarksDetector->Detect(image, rect);
            if (landmarks.size() == 0) continue;

            auto normalizedImage = mImageNormalizer->normalize(image, landmarks);

            std::vector<float> features = mFaceRecognizer->Infer(normalizedImage);

            auto classifyResult = mFaceClassifier->classify(normalizedImage, rect, landmarks, features);

            if (classifyResult == FaceClassificationResult::Pass)
            {
                eyeCoordinates.push_back(EyePair(true, true, landmarks[0]/ratio, landmarks[1]/ratio, landmarks[2]/ratio, landmarks[3]/ratio));
                templates.push_back(std::vector<float>(features.begin(), features.begin()+512));

                // DebugPrint(rect, landmarks, features);
            }
        }
        catch (const std::exception& e) {
            // Nothing to do for exceptions... move on to the next face...
            // std::cout << e.what() << std::endl;
        }
    }

    if (templates.size() > 0) {
        cv::Mat output_features = AveragePoolOnTemplates(templates);
        CvMatToTemplate(output_features, templ);
    }

    return ReturnStatus(ReturnCode::Success);
}

ReturnStatus
NullImplFRVT11::matchTemplates(
        const std::vector<uint8_t> &verifTemplate,
        const std::vector<uint8_t> &enrollTemplate,
        double &similarity)
{
    if (verifTemplate.size() == 0 || enrollTemplate.size() == 0) {
        similarity = 0;
    }
    else {
        cv::Mat f1(512, 1, CV_32F, (float *)verifTemplate.data());
        cv::Mat f2(512, 1, CV_32F, (float *)enrollTemplate.data());
        similarity = 300 * (3 - cv::norm(f1 - f2));
    }
    return ReturnStatus(ReturnCode::Success);
}

std::shared_ptr<Interface>
Interface::getImplementation()
{
    return std::make_shared<NullImplFRVT11>();
}
