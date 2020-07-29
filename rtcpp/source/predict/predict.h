#ifndef PREDICT_H
#define PREDICT_H

// includes

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>

// constants definitions

#define IMAGE_SIZE 224
#define IN_CHANNELS 3
#define OUT_CHANNELS 1
#define IN_TENSOR_SIZE IMAGE_SIZE * IMAGE_SIZE * IN_CHANNELS
#define OUT_TENSOR_SIZE IMAGE_SIZE * IMAGE_SIZE * OUT_CHANNELS

#define INPUT_NAME "image"
#define OUTPUT_NAME "segmentation"

const std::vector<int64_t> NODE_DIMS({ 1, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE });
const std::vector<const char*> INPUT_NAMES = { INPUT_NAME };
const std::vector<const char*> OUTPUT_NAMES = { OUTPUT_NAME };

const float means [3]{ 0.485f, 0.456f, 0.406f };
const float stds  [3]{ 0.229f, 0.224f, 0.225f };

// class definition

class ONNXPredictor {

public:
  cv::Mat predict_mask (cv::Mat im);
  ONNXPredictor (const void* model_data, size_t model_data_length, float thres);
  ONNXPredictor (const char* model_file, float thres);

private:
  const char* model_path;
  float thres;
  Ort::Env env;
  Ort::SessionOptions session_options;
  Ort::Session session;

  float *preprocess(cv::Mat im);
  cv::Mat postprocess(float *out);
};

#endif
