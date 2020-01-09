#include "FaceClassifier.h"
#include "SsdFaceDetector.h"
#include <fstream>

using namespace FRVT_11;

void
DumpImage(const cv::Mat& image)
{
    static int index = 0;

    cv::Mat flat = image.reshape(1, image.cols * image.rows * image.channels());
    std::vector<uchar> vec = image.isContinuous()? flat : flat.clone();

    index += 1;
    std::ofstream fout(std::string("images/image_") + std::to_string(index) + ".bin", std::ios::out | std::ios::binary);
    fout.write((char*)&vec[0], vec.size());
    fout.close();
}

float
CalcFeaturesNorm(std::vector<float> features)
{
    cv::Mat f(512, 1, CV_32F, features.data());
    float norm = cv::norm(f);
    return norm;
}

int
CalcLandmarksScale(const std::vector<int>& landmarks)
{
    std::vector<int> xPoints;
    std::vector<int> yPoints;
    for (int i = 0; i < 10; i += 2) {
        xPoints.push_back(landmarks[i]);
        yPoints.push_back(landmarks[i+1]);
    }

    int xMin = *std::min_element(xPoints.begin(), xPoints.end());
    int xMax = *std::max_element(xPoints.begin(), xPoints.end());
    int yMin = *std::min_element(yPoints.begin(), yPoints.end());
    int yMax = *std::max_element(yPoints.begin(), yPoints.end());
     
    int w = (xMax - xMin);
    int h = (yMax - yMin);

    return std::max(w, h);
}

float
CalculateLandmarksIOU(const Rect& face, const std::vector<int>& landmarks)
{
    std::vector<int> xPoints;
    std::vector<int> yPoints;
    for (int i = 0; i < 10; i += 2) {
        xPoints.push_back(landmarks[i]);
        yPoints.push_back(landmarks[i+1]);
    }

    int xMin = *std::min_element(xPoints.begin(), xPoints.end());
    int xMax = *std::max_element(xPoints.begin(), xPoints.end());
    int yMin = *std::min_element(yPoints.begin(), yPoints.end());
    int yMax = *std::max_element(yPoints.begin(), yPoints.end());

    int overlap_x0 = std::max(face.x1, xMin);
    int overlap_y0 = std::max(face.y1, yMin);
    int overlap_x1 = std::min(face.x2, xMax);
    int overlap_y1 = std::min(face.y2, yMax);

    int overlap_width = std::max(overlap_x1 - overlap_x0, 0);
    int overlap_height = std::max(overlap_y1 - overlap_y0, 0);
    int overlap_area = overlap_height * overlap_width;
    int union_area = (face.x2 - face.x1) * (face.y2 - face.y1);
    union_area += (xMax - xMin) * (yMax - yMin);
    union_area -= overlap_area;
    float iou = float(overlap_area) / union_area;

    return iou;
}

float
CalculateFaceIOU(const Rect& face)
{
    int normalization_size = 128;
    int face_overlap_x0 = std::max(face.x1, 0);
    int face_overlap_y0 = std::max(face.y1, 0);
    int face_overlap_x1 = std::min(face.x2, normalization_size);
    int face_overlap_y1 = std::min(face.y2, normalization_size);
    int face_area = (face.x2 - face.x1) * (face.y2 - face.y1);
    int face_overlap_area = std::max(face_overlap_x1 - face_overlap_x0, 0) * std::max(face_overlap_y1 - face_overlap_y0, 0);
    float iou = float(face_overlap_area) / ((normalization_size*normalization_size) + face_area - face_overlap_area);
    return iou;
}

FaceClassifier::FaceClassifier(const std::string &configDir)
{
    mFaceDetector = std::make_shared<SsdFaceDetector>(configDir, "/fd_128_509586", 128); // facessd_mobilenet_v2_dm100_swish_128x128_wider_filter20_0-509586
}

FaceClassificationResult
FaceClassifier::classify(const cv::Mat& image, const Rect& face, std::vector<int> landmarks, const std::vector<float>& features) const
{
    // std::ofstream dataFile("classification_data.txt", std::ios::out | std::ios::app);

    // dataFile << std::endl;

    float features_norm = CalcFeaturesNorm(features);
    // dataFile << "norm " << features_norm << features_norm;
    if (features_norm < mMinFeaturesNorm)
    {
        return FaceClassificationResult::Norm;
    }

    int landmarks_scale = CalcLandmarksScale(landmarks);
    // dataFile << " | lscale " << landmarks_scale;
    if (landmarks_scale < mMinLandmarksScale)
    {
        return FaceClassificationResult::Lscale;
    }

    float landmarks_iou = CalculateLandmarksIOU(face, landmarks);
    // dataFile << " | liou " << landmarks_iou;
    if (landmarks_iou < mMinLandmarksIou)
    {
        return FaceClassificationResult::Liou;
    }

    std::vector<Rect> rects = mFaceDetector->Detect(image);
    if (rects.size() == 0)
    {
        // dataFile << " | no face";
        return FaceClassificationResult::NoFace;
    }

    float face_iou = CalculateFaceIOU(rects[0]);
    // dataFile << " | fiou " << face_iou;
    if (face_iou < mMinFaceIou)
    {
        return FaceClassificationResult::Fiou;
    }

    // dataFile << "| Passed";
    // DumpImage(image);

    return FaceClassificationResult::Pass;
}
