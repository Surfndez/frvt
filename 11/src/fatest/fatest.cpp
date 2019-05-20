#include <iostream>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "frvt11.h"
#include "util.h"

using namespace FRVT;
using namespace FRVT_11;

void
SanityCheck()
{
    std::cout << "Strating sanity check..." << std::endl;

    cv::Mat image = cv::imread("/home/administrator/nist/frvt/common/images/S486-02-t10_01.ppm");
    if(!image.data) throw std::runtime_error("Could not open or find the image");

    std::cout << "Read image successfully" << std::endl;

    auto implPtr = Interface::getImplementation();

    std::cout << "Created fvrt interface" << std::endl;

    implPtr->initialize("/home/administrator/nist/frvt/11/config");
    std::cout << "Initialized implementation" << std::endl;

    std::shared_ptr<uint8_t> imageArray(new uint8_t[3 * image.rows * image.cols]);
    std::memcpy(imageArray.get(), image.data, 3 * image.rows * image.cols);

    Image imageData((uint16_t)image.cols, (uint16_t)image.rows, (uint8_t)24, imageArray, Image::Label::Unknown);
    FRVT::Multiface faces = {imageData};

    std::vector<uint8_t> templ;
    std::vector<EyePair> eyeCoordinates;

    implPtr->createTemplate(faces, TemplateRole::Enrollment_11, templ, eyeCoordinates);
}

int
main(int argc, char* argv[])
{
    SanityCheck();

	return 0;
}
