#include "predict.h"
#include <iostream>


// TODO: handle errors


MaskPredictor::MaskPredictor (const char* model_file, float thres):
                    thres(thres),
                    model{tflite::FlatBufferModel::BuildFromFile(model_file)} {
    // init interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
    interpreter->AllocateTensors();
};


cv::Mat MaskPredictor::predict_mask (cv::Mat im) {
    // Set input tensor
    int input_idx = interpreter->inputs()[0];
    float* input_tensor = interpreter->typed_tensor<float>(input_idx); 
    preprocess(im, input_tensor);

    // Invoke interpreter
    interpreter->Invoke();

    // Get output
    int output_idx = interpreter->outputs()[0];
    float *output_tensor = interpreter->typed_output_tensor<float>(output_idx);
    cv::Mat mask = postprocess(output_tensor);

    return mask;
}


void MaskPredictor::preprocess(cv::Mat im, float* input_tensor) {
    // resize and convert
    cv::resize(im, im, cvSize(IN_SIZE, IN_SIZE));

    // convert to float ptr
    uint8_t* input = im.ptr<uint8_t>(0);

    // normalize
    for (int i = 0; i < IN_TENSOR_SIZE; i++) {
        input_tensor[i] = ((input[i] / 255.0f) - 0.5f) * 2.0f; 
    }
}

cv::Mat MaskPredictor::postprocess(float *out) {
    // round using threshold
    for (int i = 0; i < OUT_TENSOR_SIZE; i++) {
        out[i] = out[i] > thres ? 0.0f : 255.0f;
    }

    // create image and convert to 3 channel image
    cv::Mat mask = cv::Mat(OUT_SIZE, OUT_SIZE, CV_32FC1, out);
    cvtColor(mask, mask, CV_GRAY2BGR);

    return mask;
}
