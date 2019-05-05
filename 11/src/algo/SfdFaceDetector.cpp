#include <cmath>
#include <algorithm>
#include <torch/script.h>

#include "SfdFaceDetector.h"

using namespace FRVT_11;

float
IOU(const Rect &box_a, const Rect &box_b)
{
    // determine the (x, y)-coordinates of the intersection rectangle
    float x_a = std::max(box_a.x1, box_b.x1);
    float y_a = std::max(box_a.y1, box_b.y1);
    float x_b = std::min(box_a.x2, box_b.x2);
    float y_b = std::min(box_a.y2, box_b.y2);

    // compute the area of intersection rectangle
    float inter_area = std::max(0.f, x_b - x_a + 1.f) * std::max(0.f, y_b - y_a + 1.f);

    // compute the area of both the prediction and ground-truth rectangles
    float box_a_area = (box_a.x2 - box_a.x1 + 1) * (box_a.y2 - box_a.y1 + 1);
    float box_b_area = (box_b.x2 - box_b.x1 + 1) * (box_b.y2 - box_b.y1 + 1);

    // compute the intersection over union by taking the intersection
    // area and dividing it by the sum of prediction + ground-truth
    // areas - the interesection area
    float iou = inter_area / float(box_a_area + box_b_area - inter_area);

    return iou;
}

Rect
Decode(const torch::Tensor &priors, const torch::Tensor &variances, const torch::Tensor &loc, float score)
{
    auto box = at::cat({priors.slice(1, 0, 2) + loc.slice(1, 0, 2) * variances[0] * priors.slice(1, 2, priors.sizes()[1]), \
                        priors.slice(1, 2, priors.sizes()[1]) * at::exp(loc.slice(1, 2, priors.sizes()[1]) * variances[1])}, 1);
    box = box[0];
    box[0] = box[0] - box[2] / 2;
    box[1] = box[1] - box[3] / 2;
    box[2] = box[2] + box[0];
    box[3] = box[3] + box[1];
    Rect rect(  int(*box[0].data<float>()),\
                int(*box[1].data<float>()),\
                int(*box[2].data<float>()),\
                int(*box[3].data<float>()),\
                score);
    return rect;
}

std::vector<Rect>
NonMaxSuppression(std::vector<Rect> &all, float threshold=0.3)
{
    if (all.size() == 0) {
        return all;
    }

    std::vector<Rect> rects;

    std::sort(std::begin(all), std::end(all), [](const Rect &a, const Rect &b) { return a.score > b.score; });

    while (all.size() > 0) {
        rects.push_back(all[0]);
        all.erase(all.begin());
        auto iter = all.begin();
        while (iter != all.end()) {
            for (const Rect& a : rects) {
                auto iou = IOU(a, *iter);
                if (iou > threshold) {
                    iter = all.erase(iter);
                }
                else {
                    ++iter;
                }
            }
        }
    }

    return rects;
}

SfdFaceDetector::SfdFaceDetector(const std::string &configDir)
{
    std::string faceDetectorModelPath = configDir + "/sfd.pt";

    // Deserialize the ScriptModule from a file using torch::jit::load().
    mFaceDetector = torch::jit::load(faceDetectorModelPath);
    assert(mFaceDetector != nullptr);
}

SfdFaceDetector::~SfdFaceDetector() {}

std::vector<Rect>
SfdFaceDetector::Detect(const ImageData &image) const
{
    std::cout << std::endl << "Detect: (" << image.height << "," << image.width << "," << image.channels << ")" << std::endl;

    // Image data to tensor
    std::vector<int64_t> sizes = {1, image.height, image.width, image.channels};
    at::TensorOptions options(at::ScalarType::Byte);
    at::Tensor tensorImage = torch::from_blob(image.data.get(), at::IntList(sizes), options);
    tensorImage = tensorImage.toType(at::kFloat);

    // Normalize pixels
    tensorImage = tensorImage - torch::tensor({104, 117, 123}, torch::requires_grad(false).dtype(torch::kFloat32));

    // HWC -> CHW
    tensorImage = tensorImage.permute({0, 3, 1, 2});

    // Execute the model and turn its output into a tensor.
    torch::jit::IValue output = mFaceDetector->forward({tensorImage});

    // Convert output to Tensors    
    c10::intrusive_ptr<c10::ivalue::Tuple> outputTuple = output.toTuple();
    auto elements = outputTuple->elements();

    std::vector<at::Tensor> tensors;
    for (int i = 0; i < elements.size(); ++i) {
        tensors.push_back(elements[i].toTensor());
    }

    // Process the output

    std::vector<Rect> rects;
    
    for (int i = 0; i < tensors.size() / 2; ++i) {
        auto t = tensors[i * 2];
        // std::cout << "t.sizes() = " << t.sizes() << std::endl;
        tensors[i * 2] = at::softmax(t, /*dim=*/1);
    }

    for (int i = 0; i < tensors.size() / 2; ++i) {
        auto ocls = tensors[i * 2][0][1];
        auto oreg = tensors[i * 2 + 1][0];
        auto stride = std::pow(2, i + 2);

        auto oclsSizes = ocls.sizes();
        for (int hindex = 0; hindex < oclsSizes[0]; ++hindex) {
            for (int windex = 0; windex < oclsSizes[1]; ++windex) {
                auto axc = stride / 2 + windex * stride;
                auto ayc = stride / 2 + hindex * stride;
                auto score = *(ocls[hindex][windex].data<float>());
                if (score > 0.5) {
                    auto loc = oreg.slice(/*dim=*/1, /*start=*/hindex, /*end=*/hindex + 1).slice(/*dim=*/2, /*start=*/windex, /*end=*/windex + 1).view({1, 4});
                    auto priors = torch::tensor({axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0}, torch::requires_grad(false).dtype(torch::kFloat32)).view({1, 4});
                    auto variances = torch::tensor({0.1, 0.2}, torch::requires_grad(false).dtype(torch::kFloat32));
                    Rect rect = Decode(priors, variances, loc, score);
                    rects.push_back(rect);
                }
            }
        }
    }

    rects = NonMaxSuppression(rects);

    for (const auto& r: rects) {
        std::cout << "Found: " << "[" << r.x1 << "," << r.y1 << "," << r.x2 << "," << r.y2 << "]" << std::endl;
    }

    return rects;
}
