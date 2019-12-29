#include <fstream>
#include <iostream>

#include "frvt11.h"
#include "TestUtils.h"
#include "ProgressBar.h"
#include "SanityTests.h"

using namespace FRVT;
using namespace FRVT_11;

/************************************/
/*********** Sanity tests ***********/
void RunSanityTests()
{
    std::cout << "Running tests..." << std::endl;
    test_similarity_calculation();
}
void test_similarity_calculation()
{
    std::cout << "\tRunnnig: test_similarity_calculation... ";

    auto implPtr = FRVT_11::Interface::getImplementation();
    std::ifstream infile("test_features_similarity.txt");
    
    float expectedScore;
    infile  >> expectedScore;

    std::vector<float> f1;
    std::vector<float> f2;

    float item;
    for (int i = 0; i < 512; i++) {
        infile >> item;
        f1.push_back(item);
    }
    for (int i = 0; i < 512; i++) {
        infile >> item;
        f2.push_back(item);
    }

    const uint8_t* bytes1 = reinterpret_cast<const uint8_t*>((const uint8_t*)f1.data());
    const uint8_t* bytes2 = reinterpret_cast<const uint8_t*>((const uint8_t*)f2.data());

    std::vector<uint8_t> templ1(512 * sizeof(float));
    std::vector<uint8_t> templ2(512 * sizeof(float));

    memcpy(templ1.data(), bytes1, 512 * sizeof(float));
    memcpy(templ2.data(), bytes2, 512 * sizeof(float));

    double score;
    implPtr->matchTemplates(templ1, templ2, score);

    std::cout << (int(score) == int(expectedScore) ? "Pass" : "Fail") << std::endl;
}
/*********** End sanity tests ***********/
/****************************************/

void
InitializeImplementation(std::shared_ptr<Interface>& implPtr)
{
    implPtr->initialize("/home/administrator/nist2/frvt/11/config");
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
    std::ofstream landmarks_file("fatest_landmarks.txt", std::ios::out | std::ios::app);
    std::ofstream features_file("fatest_features.txt", std::ios::out | std::ios::app);

    EyePair eyePair = eyeCoordinates[0];
    landmarks_file << " " << int(eyePair.xleft) << " " << int(eyePair.yleft) << " " << int(eyePair.xright) << " " << int(eyePair.yright) << std::endl;
        
    auto f = (float *)features.data();
    for (int i=0; i < 512; ++i) features_file << f[i] << " ";
    features_file << std::endl;
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

    std::vector<std::string> collectedFiles;
    std::vector<std::vector<uint8_t>> collectedFeatures;
    std::vector<std::vector<EyePair>> collectedEyes;
    
    // Extract features

    ProgressBarPrinter progress_bar("Extracting features", testList.size() / 2);
    
    for (int progress=0; progress < testList.size(); progress+=2)
    {
        if (progress == 0) progress_bar.Print(progress);

        const std::string& file = testList[progress];
        auto path = "/home/administrator/face_data/benchmarks/original/" + file;

        cv::Mat image = LoadImage(path);
        FRVT::Image imageData = CvImageToImageData(image);

        std::vector<uint8_t> features;
        std::vector<EyePair> eyeCoordinates;

        implPtr->createTemplate({imageData}, TemplateRole::Enrollment_11, features, eyeCoordinates);

        collectedFiles.push_back(file);
        collectedFeatures.push_back(features);
        collectedEyes.push_back(eyeCoordinates);

        PrintTemplate(eyeCoordinates, features);

        if (progress == 0) progress_bar.RestartTime();
        if (progress > 0) progress_bar.Print(progress / 2);
    }

    // Compare

    ProgressBarPrinter similaritiesProgressBar("Calculating similarities", GetNumComparisons(collectedFeatures.size()), 100);
    int progress = 0;

    std::vector<double> same_scores;
    std::vector<double> diff_scores;
    std::vector<double> all_scores;

    std::ofstream scores_file("fatest_scores.txt", std::ios::out);
    
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

            scores_file << collectedFiles[i] << " " << collectedFiles[j] << " " << score << std::endl;

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
    if (argc == 1) {
        std::cout << "Need test list path" << std::endl;
        return 1;
    }
    std::string listPath = argv[1];

    RunSanityTests();

    std::cout << "List path: " << listPath << std::endl;

    RunTest(listPath);

    return 0;
}
