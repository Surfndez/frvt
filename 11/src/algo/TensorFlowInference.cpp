#include "TensorFlowInference.h"

#include <iostream>

#include "TimeMeasurement.h"

extern "C"
{
    #include "c_api.h"
}

extern "C"
{
	struct TF_Tensor;
}

#include "tf_utils.hpp"

using namespace FRVT_11;

TensorFlowInference::TensorFlowInference(const std::string &modelPath, std::initializer_list<std::string> inputLayers, std::initializer_list<std::string> outputLayers) : sess(nullptr)
{
    std::string path = modelPath + ".pb";

    std::cout << "\nTensorFlow Version: " << TF_Version() << std::endl;
    std::cout << "Creating TensorFlow inference for " << path << std::endl;

    graph = tf_utils::LoadGraph(path.c_str());
    if (graph == nullptr) {
        std::cout << "Can't load graph" << std::endl;
        throw std::exception();
    }

    
}

void
TensorFlowInference::Infer(const cv::Mat& image)
{
    if (sess == nullptr)
    {
        status = TF_NewStatus();
        options = TF_NewSessionOptions();
        sess = TF_NewSession(graph, options, status);
        TF_DeleteSessionOptions(options);
    }

    TF_Output input_op = {TF_GraphOperationByName(graph, "input"), 0};
    if (input_op.oper == nullptr) {
        std::cout << "Can't init input_op" << std::endl;;
        throw std::exception();
    }

    const std::vector<std::int64_t> input_dims = {1, 128, 128, 1};
    const std::vector<float> input_vals(128*128, 0);

    TF_Tensor* input_tensor = tf_utils::CreateTensor(TF_FLOAT,
                                                    input_dims.data(), input_dims.size(),
                                                    input_vals.data(), input_vals.size() * sizeof(float));

    TF_Output out_op = {TF_GraphOperationByName(graph, "output_features"), 0};
    if (out_op.oper == nullptr) {
        std::cout << "Can't init out_op" << std::endl;
        throw std::exception();
    }

    TF_Tensor* output_tensor = nullptr;

    if (TF_GetCode(status) != TF_OK) {
        TF_DeleteStatus(status);
    }

    for (int i = 0; i < 5; ++i) {
        auto t = TimeMeasurement();
        TF_SessionRun(sess,
                    nullptr, // Run options.
                    &input_op, &input_tensor, 1, // Input tensors, input tensor values, number of inputs.
                    &out_op, &output_tensor, 1, // Output tensors, output tensor values, number of outputs.
                    nullptr, 0, // Target operations, number of targets.
                    nullptr, // Run metadata.
                    status // Output status.
                    );    
        std::cout << "TensorFlowInference::Infer "; t.Test();
    }
    // return
}
