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

            break; // should be only one face in image
        }        
    }

    // average pool on features
    // ...

    /* Note: example code, potentially not portable across machines. */
    std::vector<float> fv = {1.0, 2.0, 8.88, 765.88989};
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(fv.data());
    int dataSize = sizeof(float) * fv.size();
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
    /*
    float *featureVector = (float *)enrollTemplate.data();

    for (unsigned int i=0; i<this->featureVectorSize; i++) {
	std::cout << featureVector[i] << std::endl;
    }
    */

    similarity = rand() % 1000 + 1;
    return ReturnStatus(ReturnCode::Success);
}

std::shared_ptr<Interface>
Interface::getImplementation()
{
    return std::make_shared<NullImplFRVT11>();
}
