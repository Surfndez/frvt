#include "OpenVinoInference.h"

#include <inference_engine.hpp>

#include <iostream>
#include <chrono>
#include <ratio>
#include <thread>


using namespace FRVT_11;
using namespace InferenceEngine;

OpenVinoInference::OpenVinoInference(const std::string &modelPath) : mModelPath(modelPath)
{
    // Init();
}

void
OpenVinoInference::Init()
{
	static Core ie;
	ie.SetConfig({{ CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(NO) }}, "CPU");
	std::string xml_path = mModelPath + ".xml";
	std::string bin_path = mModelPath + ".bin";
	const file_name_t input_model_xml{ xml_path };
	const file_name_t input_model_bin{ bin_path };

	m_network_reader.ReadNetwork(fileNameToString(xml_path));
	m_network_reader.ReadWeights(fileNameToString(input_model_bin));
	m_network_reader.getNetwork().setBatchSize(1);
	bool kuku = m_network_reader.isParseSuccess();
	m_network = m_network_reader.getNetwork();


	// --------------------------- 3. Configure input & output ---------------------------------------------
	// --------------------------- Prepare input blobs -----------------------------------------------------
	InputsDataMap in_map = m_network.getInputsInfo();
	InputInfo::Ptr input_info = in_map.begin()->second;
	std::string input_name = in_map.begin()->first;

	input_info->setLayout(Layout::NHWC);
	input_info->setPrecision(Precision::FP32);

	// // 	// --------------------------- Prepare output blobs ----------------------------------------------------
	OutputsDataMap out_map = m_network.getOutputsInfo();
	DataPtr output_info = out_map.begin()->second;

	output_info->setPrecision(Precision::FP32);

	auto extension_ptr = make_so_pointer<IExtension>("libcpu_extension.so");
	ie.AddExtension(extension_ptr, "CPU");
	// -----------------------------------------------------------------------------------------------------
	// --------------------------- 4. Loading model to the plugin ------------------------------------------
    ExecutableNetwork executable_network = ie.LoadNetwork(m_network, "CPU");

	// --------------------------- 5. Create infer request -------------------------------------------------
    m_infer_request = executable_network.CreateInferRequest();
	
	// Set private members
    mInputName = input_name;
}

std::shared_ptr<InferenceEngine::Blob>
OpenVinoInference::Infer(const cv::Mat& image)
{
	if (needInit)
	{
		Init();
		needInit = false;
	}

    Blob::Ptr input = m_infer_request.GetBlob(mInputName);
    auto input_data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
    size_t channels_number = input->getTensorDesc().getDims()[1];
    size_t image_size = input->getTensorDesc().getDims()[3] * input->getTensorDesc().getDims()[2] * channels_number;
    memcpy((void*)input_data, (void*)image.data, image_size * sizeof(float));

    m_infer_request.Infer();   
    
	OutputsDataMap outputInfo(m_network.getOutputsInfo());
    Blob::Ptr outputBlob = m_infer_request.GetBlob(outputInfo.begin()->first);
	return outputBlob;
}
