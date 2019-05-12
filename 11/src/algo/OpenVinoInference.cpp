#include "OpenVinoInference.h"

#include <inference_engine.hpp>

#include "TimeMeasurement.h"

using namespace FRVT_11;
using namespace InferenceEngine;

OpenVinoInference::OpenVinoInference(const std::string &modelPath)
{
    // --------------------------- 1. Load Plugin for inference engine -------------------------------------
    mInferencePlugin = std::make_shared<InferencePlugin>(PluginDispatcher().getSuitablePlugin(TargetDevice::eCPU));

    // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
    CNNNetReader network_reader;
    network_reader.ReadNetwork(modelPath + ".xml");
    network_reader.ReadWeights(modelPath + ".bin");
    network_reader.getNetwork().setBatchSize(1);
    mCNNNetwork = std::make_shared<CNNNetwork>(network_reader.getNetwork());
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 3. Configure input & output ---------------------------------------------
    // --------------------------- Prepare input blobs -----------------------------------------------------
    InputInfo::Ptr input_info = mCNNNetwork->getInputsInfo().begin()->second;
    std::string input_name = mCNNNetwork->getInputsInfo().begin()->first;

    input_info->setLayout(Layout::NCHW);
    input_info->setPrecision(Precision::FP32);

    // --------------------------- Prepare output blobs ----------------------------------------------------
    DataPtr output_info = mCNNNetwork->getOutputsInfo().begin()->second;
    std::string output_name = mCNNNetwork->getOutputsInfo().begin()->first;

    output_info->setPrecision(Precision::FP32);
    // -----------------------------------------------------------------------------------------------------

    // Set private members
    mInputName = input_name;
    mOutputName = output_name;
}

std::shared_ptr<InferenceEngine::Blob>
OpenVinoInference::Infer(const cv::Mat& image)
{
    auto t = TimeMeasurement();

    // --------------------------- 4. Loading model to the plugin ------------------------------------------
    mExecutableNetwork = std::make_shared<ExecutableNetwork>(mInferencePlugin->LoadNetwork(*mCNNNetwork, {}));
    // -----------------------------------------------------------------------------------------------------

    std::cout << "OpenVinoInference::Infer 4. "; t.Test();

    // --------------------------- 5. Create infer request -------------------------------------------------
    mInferRequest = mExecutableNetwork->CreateInferRequestPtr();
    // -----------------------------------------------------------------------------------------------------

    std::cout << "OpenVinoInference::Infer 5. "; t.Test();
        
    // --------------------------- 6. Prepare input --------------------------------------------------------
    Blob::Ptr input = mInferRequest->GetBlob(mInputName);
    auto input_data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();

    int image_size = image.cols * image.rows;
    for (size_t pid = 0; pid < image_size; ++pid) {
        for (size_t ch = 0; ch < 1; ++ch) {
            input_data[ch * image_size + pid] = image.at<cv::Vec3b>(pid)[ch];
        }
    }
    // -----------------------------------------------------------------------------------------------------

    std::cout << "OpenVinoInference::Infer 6. "; t.Test();

    // --------------------------- 7. Do inference --------------------------------------------------------
    /* Running the request synchronously */
    mInferRequest->Infer();
    // -----------------------------------------------------------------------------------------------------

    std::cout << "OpenVinoInference::Infer 7. "; t.Test();

    // --------------------------- 8. Process output ------------------------------------------------------
    Blob::Ptr output = mInferRequest->GetBlob(mOutputName);
    // -----------------------------------------------------------------------------------------------------

    std::cout << "OpenVinoInference::Infer 8. "; t.Test();

    return output;
}
