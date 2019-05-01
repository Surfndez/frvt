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
SfdFaceDetector::Detect() const
{
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));

    // Execute the model and turn its output into a tensor.
    torch::jit::IValue output = face_detector->forward(inputs);

    // Prob the output
    std::cout << " Try to print some data... " << std::endl;
    std::cout << "isTensor: " << output.isTensor() << '\n';
    std::cout << "isBlob: " << output.isBlob() << '\n';
    std::cout << "isTuple: " << output.isTuple() << '\n';  // this returns true

    c10::intrusive_ptr<c10::ivalue::Tuple> outputTuple = output.toTuple();

    auto outputTupleElements = outputTuple->elements();

    std::cout << "Size of outputTupleElements: " << outputTupleElements.size() << std::endl;

    return std::vector<Rect>();
}
