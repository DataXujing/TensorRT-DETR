
// detr trt demo 
// xj

#include <iostream>
#include<fstream> 
#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>

#include "cuda_runtime_api.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"

#define BATCH_SIZE 1
#define INPUT_W 800
#define INPUT_H 800
#define INPUT_SIZE 800
#define NUM_CLASS 22
#define NUM_QURREY 100  //detr默认是100
#define PROB_THRESH 0.7



using namespace std;
using namespace cv;

std::vector<std::string> class_names = {"NA", "Class A", "Class B", "Class C", "Class D", "Class E", "Class F",
	"Class G", "Class H", "Class I", "Class J", "Class K", "Class L", "Class M",
	"Class N", "Class O", "Class P", "Class Q", "Class R", "Class S", "Class T","Class U"};



class Logger : public nvinfer1::ILogger
{
public:
	Logger(Severity severity = Severity::kWARNING) : reportableSeverity(severity)  //初始化参数列表
	{
	}

	void log(Severity severity, const char* msg) override
	{
		// suppress messages with severity enum value greater than the reportable
		if (severity > reportableSeverity)
			return;

		switch (severity)
		{
		case Severity::kINTERNAL_ERROR:
			std::cerr << "INTERNAL_ERROR: ";
			break;
		case Severity::kERROR:
			std::cerr << "ERROR: ";
			break;
		case Severity::kWARNING:
			std::cerr << "WARNING: ";
			break;
		case Severity::kINFO:
			std::cerr << "INFO: ";
			break;
		default:
			std::cerr << "UNKNOWN: ";
			break;
		}
		std::cerr << msg << std::endl;
	}

	Severity reportableSeverity;
};

// 这一部分可以通过trtexec实现
void onnxTotrt(const std::string& model_file, // name of the onnx model
	nvinfer1::IHostMemory** trt_model_stream, // output buffer for the TensorRT model
	Logger g_logger_,
	bool do_engine = true) {

	int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);

	// -- create the builder ------------------/
	const auto explicit_batch = static_cast<uint32_t>(BATCH_SIZE)
		<< static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(g_logger_);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicit_batch);

	// --create the parser to load onnx file---/
	auto parser = nvonnxparser::createParser(*network, g_logger_);
	if (!parser->parseFromFile(model_file.c_str(), verbosity)) {
		std::string msg("failed to parse onnx file");
		g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
		exit(EXIT_FAILURE);
	}

	// -- build the config for pass in specific parameters ---/
	builder->setMaxBatchSize(BATCH_SIZE);
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	config->setMaxWorkspaceSize(1 << 30);
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

	// std::cout <<"engine bindings dimension" << engine->getNbBindings() << std::endl;

	// -- serialize the engine,then close everything down --/
	*trt_model_stream = engine->serialize();

	//------- 序列化engine保存
	if (do_engine) {

		// serialize Model
		// IHostMemory *trt_model_stream = engine->serialize();
		std::string serialize_str;
		std::ofstream serialize_output_stream;
		serialize_str.resize((*trt_model_stream)->size());
		memcpy((void*)serialize_str.data(), (*trt_model_stream)->data(), (*trt_model_stream)->size());
		serialize_output_stream.open("./model/detr.trt");
		serialize_output_stream << serialize_str;
		serialize_output_stream.close();

	}

	parser->destroy();
	engine->destroy();
	network->destroy();
	config->destroy();
	builder->destroy();
};

//前处理
void preprocess(cv::Mat& img, float dstdata_arr[]) {

	cv::Mat img_rgb;
	cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
	cv::resize(img_rgb, img_rgb, cv::Size(INPUT_SIZE, INPUT_SIZE), cv::INTER_LINEAR);
	cv::Mat img_rgb_float;
	img_rgb.convertTo(img_rgb_float, CV_32FC3, 1 / 255.0); // 转float 归一化

	std::vector<cv::Mat> rgbChannels(3);
	std::vector<float> dstdata;
	cv::split(img_rgb_float, rgbChannels);


	for (auto i = 0; i < rgbChannels.size(); i++) {
		std::vector<float> data = std::vector<float>(rgbChannels[i].reshape(1, 1));

		for (int j = 0; j < data.size(); j++) {
			if (i == 0) {
				dstdata.push_back((data[j] - 0.485) / 0.229);
			}
			else if (i == 1) {
				dstdata.push_back((data[j] - 0.456) / 0.224);
			}
			else {
				dstdata.push_back((data[j] - 0.406) / 0.225);
			}
		}
	}

	std::copy(dstdata.begin(), dstdata.end(), dstdata_arr);

	// return dstdata_arr;
}



//后处理 

// 定义box
struct Bbox {
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	float score;
	int cid;
};

// 把box画在图像上
cv::Mat renderBoundingBox(cv::Mat image, const std::vector<Bbox> &bboxes) {
	for (auto it : bboxes) {
		float score = it.score;
		//std::cout << score;
		cv::rectangle(image, cv::Point(it.xmin, it.ymin), cv::Point(it.xmax, it.ymax), cv::Scalar(255, 204, 0), 2);
		std::string pred_class = class_names[it.cid];
		std::string label_text = pred_class + ": " + std::to_string(score);
		cv::putText(image, label_text, cv::Point(it.xmin, it.ymin-10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 204, 255));
	}
	return image;
}


// softmax
template <typename T>
int softmax(const T* src, T* dst, int length) {
	const T alpha = *std::max_element(src, src + length);
	T denominator{ 0 };

	for (int i = 0; i < length; ++i) {
		//dst[i] = std::exp(src[i] - alpha);
		dst[i] = std::exp(src[i]);

		denominator += dst[i];
	}

	for (int i = 0; i < length; ++i) {
		dst[i] /= denominator;
	}

	return 0;
}


// 后处理
vector<Bbox> postprocess(std::vector<float*> origin_output, const int &iw, const int &ih) {

	vector<Bbox> bboxes;
	Bbox bbox;

	float* Logits = origin_output[0];
	float* Boxes = origin_output[1];

	for (int i = 0; i < NUM_QURREY; i++) {
		std::vector<float> Probs;
		std::vector<float> Boxes_wh;
		for (int j = 0; j < 22; j++) {
			Probs.push_back(Logits[i * 22 + j]);
		}

		int length = Probs.size();
		std::vector<float> dst(length);

		softmax(Probs.data(), dst.data(), length);

		auto maxPosition = std::max_element(dst.begin(), dst.end() - 1);
		//std::cout << maxPosition - dst.begin() << "  |  " << *maxPosition  << std::endl;


		if (*maxPosition < PROB_THRESH) {
			Probs.clear();
			Boxes_wh.clear();
			continue;
		}
		else {
			bbox.score = *maxPosition;
			bbox.cid = maxPosition - dst.begin();

			float cx = Boxes[i * 4];
			float cy = Boxes[i * 4 + 1];
			float cw = Boxes[i * 4 + 2];
			float ch = Boxes[i * 4 + 3];

			float x1 = (cx - 0.5 * cw) * iw;
			float y1 = (cy - 0.5 * ch) * ih;
			float x2 = (cx + 0.5 * cw) * iw;
			float y2 = (cy + 0.5 * ch) * ih;

			bbox.xmin = x1;
			bbox.ymin = y1;
			bbox.xmax = x2;
			bbox.ymax = y2;

			bboxes.push_back(bbox);

			Probs.clear();
			Boxes_wh.clear();
		}
		
	}
	return bboxes;

}




float h_input[INPUT_SIZE * INPUT_SIZE * 3]; //images
float h_output_1[100 * 22];  //pred_logits
float h_output_2[100 * 4];   //pred_boxes




int main(int argc, char **argv) {

	std::string do_engine = argv[2];

	// --initial a logger
	Logger g_logger_;
	nvinfer1::IHostMemory* trt_model_stream{ nullptr };
	std::string onnx_file = "./model/detr_sim.onnx";

	// --Pass the params recorded in ONNX_file to trt_model_stream --/

	if (do_engine == "true") {

		onnxTotrt(onnx_file, &trt_model_stream, g_logger_);
		if (trt_model_stream == nullptr)
		{
			std::cerr << "Failed to load ONNX file " << std::endl;
		}

		// --deserialize the engine from the stream --- /
		nvinfer1::IRuntime* engine_runtime = nvinfer1::createInferRuntime(g_logger_);
		if (engine_runtime == nullptr)
		{
			std::cerr << "Failed to create TensorRT Runtime object." << std::endl;
		}

		// --load the infer engine -----/
		nvinfer1::ICudaEngine* engine_infer = engine_runtime->deserializeCudaEngine(trt_model_stream->data(), trt_model_stream->size(), nullptr);
		if (engine_infer == nullptr)
		{
			std::cerr << "Failed to create TensorRT Engine." << std::endl;
		}
		nvinfer1::IExecutionContext* engine_context = engine_infer->createExecutionContext();

		// --destroy stream ---/.
		trt_model_stream->destroy();
		std::cout << "loaded trt model , do inference" << std::endl;


		///////////////////////////////////////////////////////////////////
		// enqueue them up
		//////////////////////////////////////////////////////////////////

		// 加载数据，前处理

		cv::Mat image;
		image = cv::imread(argv[1], 1);

		// -- allocate host memory ------------/ 
		preprocess(image, h_input);

		//申请显存指针
		//cudaMalloc的第一个参数传递的是存储在cpu内存中的指针变量的地址，
		//cudaMalloc在执行完成后，向这个地址中写入了一个地址值（此地址值是GPU显存里的）
		void* buffers[3];
		cudaMalloc(&buffers[0], INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float));  //<- inputs
		cudaMalloc(&buffers[1], 100 * 22 * sizeof(float)); //<- pred_logits
		cudaMalloc(&buffers[2], 100 * 4 * sizeof(float)); //<- pred_boxes



		// cudaMemcpy用于在主机（Host）和设备（Device）之间往返的传递数据，用法如下：

		// 主机到设备：cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice)
		// 设备到主机：cudaMemcpy(h_A,d_A,nBytes,cudaMemcpyDeviceToHost)
		// 注意：该函数是同步执行函数，在未完成数据的转移操作之前会锁死并一直占有CPU进程的控制权，所以不用再添加cudaDeviceSynchronize()函数
		cudaMemcpy(buffers[0], h_input, INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float), cudaMemcpyHostToDevice);

		// -- do execute --------//
		// int16_t, int32_t..., 等， 使用typedef facility定义特定大小intergers在不同的机器上， 并提供了代码可移植性。s
		int32_t BATCH_SIZE_ = 1;
		//engine_context->execute(BATCH_SIZE_, buffers);
		engine_context->executeV2(buffers);


		cudaMemcpy(h_output_1, buffers[1],100 * 22 * sizeof(float),cudaMemcpyDeviceToHost);
		cudaMemcpy(h_output_2, buffers[2], 100 * 4 * sizeof(float), cudaMemcpyDeviceToHost);




		std::cout << "开始打印TensorRT返回的结果：" << std::endl;
		std::vector<float*> output = { h_output_1 ,h_output_2 };

		// 后处理
		vector<Bbox> bboxes = postprocess(output, image.cols, image.rows);

		cv::Mat showImage;
		showImage = renderBoundingBox(image, bboxes);
		cv::imwrite("res.jpg", showImage);


		cudaFree(buffers[0]);
		cudaFree(buffers[1]);
		cudaFree(buffers[2]);

		//engine_runtime->destroy();
		//engine_infer->destroy();


	}
	else {

		// 如果基于序列化的engine,直接在engine文件中反序列化
		nvinfer1::IRuntime* engine_runtime = nvinfer1::createInferRuntime(g_logger_);
		std::string cached_path = "./model/detr.trt";
		std::ifstream fin(cached_path);
		std::string cached_engine = "";
		while (fin.peek() != EOF) {
			std::stringstream buffer;
			buffer << fin.rdbuf();
			cached_engine.append(buffer.str());
		}
		fin.close();
		nvinfer1::ICudaEngine* engine_infer = engine_runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
		int num_index = engine_infer->getNbBindings();
		int input_index = engine_infer->getBindingIndex("inputs"); //1x3x800 X 800
		//std::string input_name = engine_infer->getBindingName(0)
		int output_index_1 = engine_infer->getBindingIndex("pred_logits"); 
		int output_index_2 = engine_infer->getBindingIndex("pred_boxes");

		nvinfer1::IExecutionContext* engine_context = engine_infer->createExecutionContext();

		if (engine_context == nullptr)
		{
			std::cerr << "Failed to create TensorRT Execution Context." << std::endl;
		}

		// cached_engine->destroy();
		std::cout << "loaded trt model , do inference" << std::endl;


		///////////////////////////////////////////////////////////////////
		// enqueue them up
		//////////////////////////////////////////////////////////////////

		// 加载数据，前处理
		cv::Mat image;
		image = cv::imread(argv[1], 1);

		// -- allocate host memory ------------/ 

		preprocess(image, h_input);
		//image.release();


		//申请显存指针
		//cudaMalloc的第一个参数传递的是存储在cpu内存中的指针变量的地址，
		//cudaMalloc在执行完成后，向这个地址中写入了一个地址值（此地址值是GPU显存里的）
		void* buffers[3];
		cudaMalloc(&buffers[0], INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float));  //<- inputs
		cudaMalloc(&buffers[1], 100 * 22 * sizeof(float)); //<- pred_logits
		cudaMalloc(&buffers[2], 100 * 4 * sizeof(float)); //<- pred_boxes

		// cudaMemcpy用于在主机（Host）和设备（Device）之间往返的传递数据，用法如下：

		// 主机到设备：cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice)
		// 设备到主机：cudaMemcpy(h_A,d_A,nBytes,cudaMemcpyDeviceToHost)
		// 注意：该函数是同步执行函数，在未完成数据的转移操作之前会锁死并一直占有CPU进程的控制权，所以不用再添加cudaDeviceSynchronize()函数
		cudaMemcpy(buffers[0], h_input, INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float), cudaMemcpyHostToDevice);

		// -- do execute --------//
		// int16_t, int32_t..., 等， 使用typedef facility定义特定大小intergers在不同的机器上， 并提供了代码可移植性。s
		int32_t BATCH_SIZE_ = 1;
		//engine_context->execute(BATCH_SIZE_, buffers);
		engine_context->executeV2(buffers);


		cudaMemcpy(h_output_1, buffers[1], 100 * 22 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_output_2, buffers[2], 100 * 4 * sizeof(float), cudaMemcpyDeviceToHost);

		std::cout << "开始打印TensorRT返回的结果：" << std::endl;
		std::vector<float*> output = { h_output_1 ,h_output_2 };

		// 后处理
		vector<Bbox> bboxes = postprocess(output, image.cols, image.rows);

		std::cout << "后处理完成！" << std::endl;


		cv::Mat showImage;
		showImage = renderBoundingBox(image, bboxes);
		cv::imwrite("res.jpg", showImage);


		cudaFree(buffers[0]);
		cudaFree(buffers[1]);
		cudaFree(buffers[2]);



		//engine_runtime->destroy();
		//engine_infer->destroy();


	}

	// // cudaStreamDestroy(stream);


	return 0;
}
