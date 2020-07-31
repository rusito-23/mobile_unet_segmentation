#ifndef PREDICT_H
#define PREDICT_H

// includes

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"

// constants

#define IN_SIZE 224
#define IN_CHANNELS 3
#define IN_TENSOR_SIZE IN_SIZE * IN_SIZE * IN_CHANNELS

#define OUT_SIZE 112
#define OUT_CHANNELS 1
#define OUT_TENSOR_SIZE OUT_SIZE * OUT_SIZE * OUT_CHANNELS

// mask predictor

class MaskPredictor {

public:
    cv::Mat predict_mask (cv::Mat im);
    MaskPredictor(const char* model_file, float thres);

private:
    float thres;
    std::unique_ptr<tflite::Interpreter> interpreter;
    std::unique_ptr<tflite::FlatBufferModel> model;

    void preprocess(cv::Mat im, float* input_tensor);
    cv::Mat postprocess(float *out);
};

#endif
