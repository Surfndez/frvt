#include <algorithm>

#include <torch/script.h>
#include <opencv2/core.hpp>

#include "FanLandmarksDetector.h"

using namespace FRVT_11;

int REFERENCE_SCALE = 195;

std::vector<int>
Transform(std::vector<float> point, float* center, float scale, float resolution, bool invert=false)
{
    std::cout << "Transform 0" << std::endl;

    cv::Mat _pt = cv::Mat::ones(3, 1, CV_32FC1);
    _pt.at<float>(0, 0) = point[0];
    _pt.at<float>(1, 0) = point[1];

    auto h = 200.0 * scale;
    cv::Mat t = cv::Mat::eye(3, 3, CV_32FC1);
    t.at<float>(0, 0) = resolution / h;
    t.at<float>(1, 1) = resolution / h;
    t.at<float>(0, 2) = resolution * (-center[0] / h + 0.5);
    t.at<float>(1, 2) = resolution * (-center[1] / h + 0.5);

    if (invert)
        t = t.inv();

    cv::Mat new_point = t * _pt;

    return {int(new_point.at<float>(0, 0)), int(new_point.at<float>(1, 0))};
}

cv::Mat
CropImage(const cv::Mat &image, float* center, float scale , float resolution=256.0f)
{
    std::cout << "CropImage 0" << std::endl;

    // Crop around the center point
    /* Crops the image around the center. Input is expected to be an np.ndarray */
    auto ul = Transform({1, 1}, center, scale, resolution, true);
    auto br = Transform({resolution, resolution}, center, scale, resolution, true);

    std::cout << "CropImage 1" << std::endl;

    cv::Mat newImg = cv::Mat::zeros(br[1] - ul[1], br[0] - ul[0], CV_8UC3);

    std::cout << "CropImage 2" << std::endl;

    auto ht = image.rows;
    auto wd = image.cols;

    std::vector<int> newX = {std::max(1, -ul[0] + 1), std::min(br[0], wd) - ul[0]};
    std::vector<int> newY = {std::max(1, -ul[1] + 1), std::min(br[1], ht) - ul[1]};

    std::vector<int> oldX = {std::max(1, ul[0]) + 1, std::min(br[0], wd)};
    std::vector<int> oldY = {std::max(1, ul[1]) + 1, std::min(br[1], ht)};

    std::cout << "CropImage 3 " << oldX[0] << "," << oldY[0] << "," << oldX[1] << "," << oldY[1] << std::endl;
    cv::Mat oldImg = image(cv::Range(oldY[0] - 1, oldY[1]), cv::Range(oldX[0] - 1, oldX[1]));
    std::cout << "CropImage 4 " << newX[0] << "," << newY[0] << "," << newX[1] << "," << newY[1] << std::endl;
    oldImg.copyTo(newImg(cv::Rect(newX[0] - 1, newY[0] - 1, newX[1] - newX[0] + 1, newY[1] - newY[0] + 1)));

    return newImg;
}

FanLandmarksDetector::FanLandmarksDetector(const std::string &configDir)
{
    std::string landmarksDetectorModelPath = configDir + "/fan.pt";

    // Deserialize the ScriptModule from a file using torch::jit::load().
    mLandmarksDetector = torch::jit::load(landmarksDetectorModelPath);
    assert(mLandmarksDetector != nullptr);
}

FanLandmarksDetector::~FanLandmarksDetector() {}

std::vector<int>
FanLandmarksDetector::Detect(const ImageData& imageData, const Rect &face) const
{
    std::cout << "Detect landmarks... " << std::endl;

    cv::Mat image(imageData.height, imageData.width, CV_8UC3, imageData.data.get());

    // calculate center and scale
    int d[] = {face.x1, face.y1, face.x2, face.y2};
    float center[] = {d[2] - (d[2] - d[0]) / 2.0f, d[3] - (d[3] - d[1]) / 2.0f};
    center[1] = center[1] - (d[3] - d[1]) * 0.12f;
    float scale = (d[2] - d[0] + d[3] - d[1]) / REFERENCE_SCALE;

    // Crop image
    image = CropImage(image, center, scale);

    // create image tensor
    std::vector<int64_t> sizes = {1, image.rows, image.cols, image.channels()};
    at::TensorOptions options(at::ScalarType::Byte);
    at::Tensor tensorImage = torch::from_blob(image.data, at::IntList(sizes), options);
    tensorImage = tensorImage.toType(at::kFloat);

    // HWC -> CHW
    tensorImage = tensorImage.permute({0, 3, 1, 2});

    // Normalize image
    tensorImage = tensorImage.div(255.0);

    // Inference
    torch::jit::IValue output = mLandmarksDetector->forward({tensorImage});

    // Adjust output
    //pts, pts_img = get_preds_fromhm(out, center, scale)
    //pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
    //infer_detection(detections[0])

    std::cout << "Done!" << std::endl;

    std::vector<int> landmarks;
    return landmarks;
}
