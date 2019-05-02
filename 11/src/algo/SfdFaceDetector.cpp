#include <cmath>
#include <torch/script.h>

#include "SfdFaceDetector.h"

using namespace FRVT_11;

SfdFaceDetector::SfdFaceDetector(const std::string &configDir)
{
    std::string face_detector_model_path = configDir + "/sfd.pt";

    // Deserialize the ScriptModule from a file using torch::jit::load().
    face_detector = torch::jit::load(face_detector_model_path);
    assert(face_detector != nullptr);
}

SfdFaceDetector::~SfdFaceDetector() {}

std::vector<Rect>
SfdFaceDetector::Detect(std::shared_ptr<uint8_t> data, int width, int height, int channels) const
{
    std::cout << "Detect: (" << height << "," << width << "," << channels << ")" << std::endl;

    std::vector<uint8_t> image;

    // Normalize pixels
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int base_index = r * c * channels;
            image.push_back(data.get()[base_index + 0] - 104);
            image.push_back(data.get()[base_index + 1] - 117);
            image.push_back(data.get()[base_index + 2] - 123);
        }
    }

    // Image data to tensor
    std::vector<int64_t> sizes = {1, channels, height, width};
    at::TensorOptions options(at::ScalarType::Byte);
    at::Tensor tensor_image = torch::from_blob(image.data(), at::IntList(sizes), options);
    tensor_image = tensor_image.toType(at::kFloat);

    // Resize image for faster inference
    tensor_image = tensor_image.resize_({1, 3, int(height * 0.5), int(width * 0.5)}); // TODO: !!! This probably shouldn't stay here. Detection won't be so good...

    // Execute the model and turn its output into a tensor.
    torch::jit::IValue output = face_detector->forward({tensor_image});

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

        // std::cout << ocls << std::endl;

        auto oclsSizes = ocls.sizes();
        for (int hindex = 0; hindex < oclsSizes[0]; ++hindex) {
            for (int windex = 0; windex < oclsSizes[1]; ++windex) {
                auto axc = stride / 2 + windex * stride;
                auto ayc = stride / 2 + hindex * stride;
                auto score = *(ocls[hindex][windex].data<float>());
                if (score > 0.05) {
                    auto loc = oreg.slice(/*dim=*/1, /*start=*/hindex, /*end=*/hindex + 1).slice(/*dim=*/2, /*start=*/windex, /*end=*/windex + 1).view({1, 4});
                    auto priors = torch::tensor({axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0}, torch::requires_grad(false).dtype(torch::kFloat32)).view({1, 4});
                    auto variances = torch::tensor({0.1, 0.2}, torch::requires_grad(false).dtype(torch::kFloat32));

                    // decode
                    auto box = at::cat({priors.slice(1, 0, 2) + loc.slice(1, 0, 2) * variances[0] * priors.slice(1, 2, priors.sizes()[1]), \
                                        priors.slice(1, 2, priors.sizes()[1]) * at::exp(loc.slice(1, 2, priors.sizes()[1]) * variances[1])}, 1);
                    box = box[0];

                    Rect rect(  int(*(box[0] - box[2] / 2).data<float>()),\
                                int(*(box[1] - box[3] / 2).data<float>()),\
                                int(*(box[2] + box[0]).data<float>()),\
                                int(*(box[3] + box[1]).data<float>()),\
                                score);

                    rects.push_back(rect);
                }
            }
        }
    }

    for (const auto& r: rects) {
        std::cout << "Found: " << "[" << r.x1 << "," << r.y1 << "," << r.x2 << "," << r.y2 << "]" << std::endl;
    }

    return rects;
}
