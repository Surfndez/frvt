#include <fstream>
#include <iostream>

#include "frvt11.h"
#include "TestUtils.h"
#include "ProgressBar.h"
#include <opencv2/opencv.hpp>


using namespace FRVT;
using namespace FRVT_11;

#include <iostream>
#include <chrono>
#include <ratio>
#include <thread>

void
InitializeImplementation(std::shared_ptr<Interface>& implPtr)
{
    implPtr->initialize("/home/administrator/nist/frvt/11/config");
}

Image
CvImageToImageData(const cv::Mat& image)
{
    std::shared_ptr<uint8_t> imageArray(new uint8_t[3 * image.rows * image.cols]);
    std::memcpy(imageArray.get(), image.data, 3 * image.rows * image.cols);

    Image imageData((uint16_t)image.cols, (uint16_t)image.rows, (uint8_t)24, imageArray, Image::Label::Unknown);
    
    return imageData;
}

void
PrintTemplate(const std::vector<EyePair>& eyeCoordinates, const std::vector<uint8_t>& features)
{
    EyePair eyePair = eyeCoordinates[0];
        
    std::cout << "Test output: " << std::endl;
    std::cout << "\t" << int(eyePair.xleft) << " " << int(eyePair.yleft) << " " << int(eyePair.xright) << " " << int(eyePair.yright) << std::endl;
    auto f = (float *)features.data();
    std::cout << "\t";
    for (int i=0; i < 10; ++i)
        std::cout << f[i] << " ";
    std::cout << std::endl;

    // std::ofstream outFile("/home/administrator/nist/debug/features.txt");
    // for (int i=0; i < 512; i++) outFile << f[i] << "\n";
}

int
GetNumComparisons(int numFeatures)
{
    int comps = 0;
    for (int i = numFeatures; i > 0; i--)
    {
        comps += i;
    }
    return comps;
}


void
RunTest(const std::string& list_path)
{
    // Load test list

    auto testList = ReadTestList(list_path);
    // std::vector<std::string> testList(testList_.begin(), testList_.begin()+100);

    // Create FRVT implementation

    auto implPtr = FRVT_11::Interface::getImplementation();
    InitializeImplementation(implPtr);

    // Collections

    std::vector<std::vector<uint8_t>> collectedFeatures;
    std::vector<std::vector<EyePair>> collectedEyes;

    
    // Extract features

    ProgressBarPrinter progress_bar("Extracting features", testList.size() / 2);
    
    for (int progress=0; progress < testList.size(); progress+=2)
    {
        if (progress == 0) progress_bar.Print(progress);

        const std::string& file = testList[progress];
        auto path = "/home/administrator/face_data/benchmarks/original" + file;

        cv::Mat image = LoadImage(path);
        FRVT::Image imageData = CvImageToImageData(image);


       std::vector<uint8_t> features;
        std::vector<EyePair> eyeCoordinates;

        implPtr->createTemplate({imageData}, TemplateRole::Enrollment_11, features, eyeCoordinates);

        collectedFeatures.push_back(features);
        collectedEyes.push_back(eyeCoordinates);

        // PrintTemplate(eyeCoordinates, features);

        if (progress == 0) progress_bar.RestartTime();
        if (progress > 0) progress_bar.Print(progress / 2);
    }

    // Compare

    ProgressBarPrinter similaritiesProgressBar("Calculating similarities", GetNumComparisons(collectedFeatures.size()), 100);
    int progress = 0;

    std::vector<double> same_scores;
    std::vector<double> diff_scores;
    std::vector<double> all_scores;
    
    for (int i = 0; i < collectedFeatures.size(); i++)
    {
        auto features1 = collectedFeatures[i];
        for (int j = i + 1; j < collectedFeatures.size(); j++)
        {
            if (progress == 0) similaritiesProgressBar.Print(progress);

            auto features2 = collectedFeatures[j];
            double score = 0;
            implPtr->matchTemplates(features1, features2, score);

            bool isSame = testList[i * 2 + 1] == testList[j * 2 + 1];
            if (isSame) same_scores.push_back(score);
            else diff_scores.push_back(score);
            all_scores.push_back(score);

            if (progress == 0) similaritiesProgressBar.RestartTime();
            if (progress > 0) similaritiesProgressBar.Print(progress);
            progress++;
        }
    }

    // Output TPR

    std::cout << std::endl;
    std::cout << "TPR @ FPR 1:" << 10 << " = " << CalculateTPR(10, diff_scores, same_scores) << std::endl;
    std::cout << "TPR @ FPR 1:" << 100 << " = " << CalculateTPR(100, diff_scores, same_scores) << std::endl;
    std::cout << "TPR @ FPR 1:" << 1000 << " = " << CalculateTPR(1000, diff_scores, same_scores) << std::endl;
}


int
main(int argc, char* argv[])
{
    char *s = "TF_CPP_MIN_LOG_LEVEL=3";
    putenv(s); // Disable TensorFlow logs
     char *a = "OMP_NUM_THREADS=1";
    putenv(a); // Disable MKL muilti-threading

    if (argc == 1) {
        std::cout << "Need test list path" << std::endl;
        return 1;
    }
    std::string listPath = argv[1];

    std::cout << "List path: " << listPath << std::endl;

    RunTest(listPath);

    return 0;
}
