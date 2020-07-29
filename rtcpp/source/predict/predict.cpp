#include "predict.h"
#include <iostream>

#define FOREACH_PIXEL(channels) for (int c = 0; c < channels; ++c) { \
                                for (int h = 0; h < IMAGE_SIZE; ++h) { \
                                for (int w = 0; w < IMAGE_SIZE; ++w) { \
                                int old_index = IMAGE_SIZE*channels*h + w*channels + c; \
                                int new_index = c*IMAGE_SIZE*IMAGE_SIZE + h*IMAGE_SIZE + w; \

#define ENDFOR_PIXEL }}}

//*************************************************************************
// Initialization
//*************************************************************************

ONNXPredictor::ONNXPredictor (const void* model_data,
                              size_t model_data_length,
                              float thres):
      env(ORT_LOGGING_LEVEL_WARNING, "test"),
      session(env, model_data, model_data_length, session_options),
      thres(thres) {
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

ONNXPredictor::ONNXPredictor (const char* model_file, float thres):
      env(ORT_LOGGING_LEVEL_WARNING, "test"),
      session(env, model_file, session_options),
      thres(thres) {
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

//*************************************************************************
// Mask prediction
//*************************************************************************

cv::Mat ONNXPredictor::predict_mask (cv::Mat im) {
  //*************************************************************************
  // Preprocess image
  float* input = preprocess(im);

  //*************************************************************************
  // Convert vector to input tensor
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                  input,
                                                  IN_TENSOR_SIZE,
                                                  NODE_DIMS.data(), 4);

  //*************************************************************************
  // Pass input tensor through the net
  auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                    INPUT_NAMES.data(),
                                    &input_tensor, 1,
                                    OUTPUT_NAMES.data(), 1);

  //*************************************************************************
  // Convert output to image
  float* output = output_tensors.front().GetTensorMutableData<float>();
  cv::Mat mask = postprocess(output);

  return mask;
}

//*************************************************************************
// Image Processing
// TODO: normalize + denormalize images
//*************************************************************************

float* ONNXPredictor::preprocess(cv::Mat im) {
  // resize and convert to 32 bit floats
  cv::resize(im, im, cvSize(IMAGE_SIZE, IMAGE_SIZE));
  im.convertTo(im, CV_32F);
  cvtColor(im, im, CV_BGR2RGB);

  // fix channels
  float* im_content = reinterpret_cast<float*>(im.ptr());
  float* input = new float[IMAGE_SIZE*IMAGE_SIZE*IN_CHANNELS];
  FOREACH_PIXEL(IN_CHANNELS)
    // normalize pixels
    input[new_index] = im_content[old_index];
  ENDFOR_PIXEL

  return input;
}

cv::Mat ONNXPredictor::postprocess(float *out) {
  // fix channels
  float* output = new float[IMAGE_SIZE*IMAGE_SIZE*OUT_CHANNELS];
  FOREACH_PIXEL(OUT_CHANNELS)
    float new_pixel = out[old_index] > thres ? 0 : 255;
    output[new_index] = new_pixel;
  ENDFOR_PIXEL

  // create image and convert to 3 channel image
  cv::Mat mask = cv::Mat(IMAGE_SIZE, IMAGE_SIZE, CV_32FC1, output);
  cvtColor(mask, mask, CV_GRAY2BGR);

  return mask;
}
