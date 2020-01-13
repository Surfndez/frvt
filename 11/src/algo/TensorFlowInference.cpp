#include "TensorFlowInference.h"

#include <iostream>
#include <stdexcept>

#include "c_api.h"
#include "tf_utils.hpp"

extern "C"
{
	struct TF_Tensor;
}

using namespace FRVT_11;

TensorFlowInference::TensorFlowInference(const std::string &modelPath, std::initializer_list<std::string> inputLayers, std::initializer_list<std::string> outputLayers) :
    sess(nullptr), mInputLayers(inputLayers), mOutputLayers(outputLayers)
{
    std::string path = modelPath + ".pb";

    graph = tf_utils::LoadGraph(path.c_str());
    if (graph == nullptr) {
        std::cout << "Can't load graph" << std::endl;
        throw std::exception();
    } 
}

void
TensorFlowInference::Init()
{
    if (sess == nullptr)
    {
        status = TF_NewStatus();
        options = TF_NewSessionOptions();

        uint8_t config[4] = {0x10, 0x1, 0x28, 0x1};
        TF_SetConfig(options, (void*)config, 4, status);
        if (TF_GetCode(status) != TF_OK) std::runtime_error("Failed to set number of threads");

        sess = TF_NewSession(graph, options, status);
        TF_DeleteSessionOptions(options);
    }
}

std::vector<TensorFlowInference::deleted_unique_ptr<TF_Tensor>>
TensorFlowInference::Infer(const cv::Mat& image)
{
    Init();

    // Prapare inputs

    auto element_size = image.type() == CV_32F || image.type() == CV_32FC3 ? sizeof(float) : sizeof(char);
    auto tf_type = image.type() == CV_32F || image.type() == CV_32FC3 ? TF_FLOAT : TF_UINT8;

    TF_Output input_op = {TF_GraphOperationByName(graph, mInputLayers[0].c_str()), 0}; // Currently assume single input...
    if (input_op.oper == nullptr) {
        throw std::runtime_error("Can't init input_op");
    }

    const std::vector<std::int64_t> input_dims = {1, image.rows, image.cols, image.channels()};
    auto input_size = image.rows * image.cols * image.channels() * element_size;

    TF_Tensor* input_tensor = tf_utils::CreateTensor(tf_type, input_dims.data(), input_dims.size(), image.data, input_size);

    // Prepare outputs

    std::vector<TF_Output> outputs;

    for (std::string layer_name : mOutputLayers)
    {
        // Setup graph outputs
        TF_Operation* output_op = TF_GraphOperationByName(graph, layer_name.c_str());

        if (output_op != nullptr) outputs.push_back({ output_op, 0 });
        else throw std::runtime_error("Output Layer " + layer_name + " Not found in model");
    }

    std::vector<TF_Tensor*> output_values(outputs.size());

    // Perform inference
    TF_SessionRun(sess,
                nullptr, // Run options.
                &input_op, &input_tensor, 1, // Input tensors, input tensor values, number of inputs.
                outputs.data(), output_values.data(), outputs.size(), // Output tensors, output tensor values, number of outputs.
                nullptr, 0, // Target operations, number of targets.
                nullptr, // Run metadata.
                status // Output status.
                );
    
    if (TF_GetCode(status) != TF_Code::TF_OK)
    {
        std::string errMsg(TF_Message(status));
        throw std::runtime_error("Error from TF_SessionRun: " + errMsg);
    }

    // Return result

    std::vector<deleted_unique_ptr<TF_Tensor>> return_values(outputs.size());

    for (int i = 0; i < output_values.size(); i++)
    {
        return_values[i] = deleted_unique_ptr<TF_Tensor>(output_values[i], [](TF_Tensor* p)
        {
            if (p != nullptr) TF_DeleteTensor(p);
        });
    }
    TF_DeleteTensor(input_tensor);

    return return_values;
}