#ifndef PREDICT_H
#define PREDICT_H

// includes

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#pragma clang pop

// constants

#define IN_SIZE 224
#define IN_CHANNELS 3
#define IN_TENSOR_SIZE IN_SIZE * IN_SIZE * IN_CHANNELS

#define OUT_SIZE 112
#define OUT_CHANNELS 1
#define OUT_TENSOR_SIZE OUT_SIZE * OUT_SIZE * OUT_CHANNELS

/**
 Mask Predictor
 Handles the communication with the Tensorflow Lite API
 and the pre/post-processing needed for the given OpenCV Mat.
 */
class MaskPredictor {
public:

    /**
     Constructor
     
     @param model_file String indicating the `tflite` model path.
     @param thres Threshold used to compute the B&W mask from the given probabilites.
     */
    MaskPredictor(const char* model_file, float thres);

    /**
     Predict a mask.
     
     Pre-processes a given mat and passes it through the net.
     Then, post-processes the given probabilities into a Mat.
     
     @param im OpenCV Mat - BGR - 8UC3
     @returns Mask in OpenCV Mat format - BGR - 8UC3 (contains 0's and 255's only)
     */
    cv::Mat predict_mask (cv::Mat im);

private:
    float thres;
    std::unique_ptr<tflite::Interpreter> interpreter;
    std::unique_ptr<tflite::FlatBufferModel> model;

    void preprocess(cv::Mat im, float* input_tensor);
    cv::Mat postprocess(float *out);
};

#endif
