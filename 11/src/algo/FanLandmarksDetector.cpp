#include <algorithm>

#include <torch/script.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "FanLandmarksDetector.h"

using namespace FRVT_11;

int REFERENCE_SCALE = 195;

std::vector<int>
Transform(std::vector<float> point, float* center, float scale, float resolution, bool invert=false)
{
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
CropImage(const cv::Mat &image, float* center, float scale, float resolution=256.0f)
{
    // Crop around the center point
    /* Crops the image around the center. Input is expected to be an np.ndarray */
    auto ul = Transform({1, 1}, center, scale, resolution, true);
    auto br = Transform({resolution, resolution}, center, scale, resolution, true);

    cv::Mat newImg = cv::Mat::zeros(br[1] - ul[1], br[0] - ul[0], CV_8UC3);

    auto ht = image.rows;
    auto wd = image.cols;

    std::vector<int> newX = {std::max(1, -ul[0] + 1), std::min(br[0], wd) - ul[0]};
    std::vector<int> newY = {std::max(1, -ul[1] + 1), std::min(br[1], ht) - ul[1]};

    std::vector<int> oldX = {std::max(1, ul[0]) + 1, std::min(br[0], wd)};
    std::vector<int> oldY = {std::max(1, ul[1]) + 1, std::min(br[1], ht)};

    cv::Mat oldImg = image(cv::Range(oldY[0] - 1, oldY[1]), cv::Range(oldX[0] - 1, oldX[1]));
    oldImg.copyTo(newImg(cv::Rect(newX[0] - 1, newY[0] - 1, newX[1] - newX[0] + 1, newY[1] - newY[0] + 1)));

    cv::resize(newImg, newImg, cv::Size(resolution, resolution), 0, 0, cv::INTER_LINEAR);

    return newImg;
}

std::vector<int>
DecodeOutput(at::Tensor &outputTensor, float* center, float scale)
{
    auto res = outputTensor.view({outputTensor.size(0), outputTensor.size(1), outputTensor.size(2) * outputTensor.size(3)}).max(2);
    auto maxValue = std::get<0>(res);
    auto idx = std::get<1>(res);
    idx = idx + 1;

    auto preds = idx.view({idx.size(0), idx.size(1), 1}).repeat({1, 1, 2});
    preds = at::_cast_Float(preds);
    for (int i = 0; i < preds.size(1); ++i) {
        preds[0][i][0] = (preds[0][i][0] - 1) % outputTensor.size(3) + 1;
        preds[0][i][1] = preds[0][i][1].add_(-1).div_(outputTensor.size(2)).floor_().add_(1);
    }

    std::cout << "0" << std::endl;

    for (int i = 0; i < preds.size(0); ++i) {
        for (int j = 0; j < preds.size(1); ++j) {
            int pX = int(preds[i][j][0].item<int>()) - 1;
            int pY = int(preds[i][j][1].item<int>()) - 1;
            if (pX > 0 && pX < 63 && pY > 0 && pY < 63) {
                std::vector<at::Tensor> diff = {
                    outputTensor[i][j][pY][pX + 1] - outputTensor[i][j][pY][pX - 1],
                    outputTensor[i][j][pY + 1][pX] - outputTensor[i][j][pY - 1][pX]};
                preds[i][j][0] = diff[0].add_(diff[0].sign_().mul_(.25));
                preds[i][j][1] = diff[1].add_(diff[0].sign_().mul_(.25));
            }
        }
    }

    preds = preds.add_(-.5);

    auto preds_orig = at::zeros(preds.sizes());
    for (int i = 0; i < outputTensor.size(0); ++i) {
        for (int j = 0; j < outputTensor.size(1); ++j) {
            std::vector<float> point = {preds[i][j][0].item<float>(), preds[i][j][1].item<float>()};
            auto transformed = Transform(point, center, scale, outputTensor.size(2), true);
            preds_orig[i][j][0] = transformed[0];
            preds_orig[i][j][1] = transformed[1];
        }
    }
    
    preds_orig = preds_orig.view({68, 2});

    std::cout << "preds_orig[30] = " << preds_orig[30][0].item<int>() << "," << preds_orig[30][1].item<int>() << std::endl;
    std::cout << "preds_orig[48] = " << preds_orig[48][0].item<int>() << "," << preds_orig[48][1].item<int>() << std::endl;
    std::cout << "preds_orig[54] = " << preds_orig[54][0].item<int>() << "," << preds_orig[54][1].item<int>() << std::endl;

    std::vector<int> landmarks;
    

    return landmarks;
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

    // Convert output to Tensors    
    c10::intrusive_ptr<c10::ivalue::Tuple> outputTuple = output.toTuple();
    at::Tensor outputTensor = outputTuple->elements()[outputTuple->elements().size()-1].toTensor();

    // Adjust output
    auto landmarks = DecodeOutput(outputTensor, center, scale);
    //pts, pts_img = get_preds_fromhm(out, center, scale)
    //pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
    //infer_detection(detections[0])

    std::cout << "Done!" << std::endl;

    return landmarks;
}
