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
SfdFaceDetector::Detect(std::shared_ptr<uint8_t> data, int width, int height, int depth) const
{
    std::vector<uint8_t> image;

    // Normalize pixels
    for (int r = 0; r < height; ++r)
        for (int c = 0; c < width; ++c) {
            int base_index = r * c * 3;
            image.push_back(data.get()[base_index + 0] - 104);
            image.push_back(data.get()[base_index + 1] - 117);
            image.push_back(data.get()[base_index + 2] - 123);
        }

    std::vector<int64_t> sizes = {1, 3, height, width};
    at::TensorOptions options(at::ScalarType::Byte);
    at::Tensor tensor_image = torch::from_blob(image.data(), at::IntList(sizes), options);
    tensor_image = tensor_image.toType(at::kFloat);

    // Execute the model and turn its output into a tensor.
    torch::jit::IValue output = face_detector->forward({tensor_image});

    // Process the output
    c10::intrusive_ptr<c10::ivalue::Tuple> outputTuple = output.toTuple();
    
    auto outputTupleElements = outputTuple->elements();

    std::cout << "Size of outputTupleElements: " << outputTupleElements.size() << std::endl;

    return std::vector<Rect>();
}
