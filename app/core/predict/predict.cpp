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
    // Preprocess image
    float* input = preprocess(im);

    // Set input tensor
    int input_idx = interpreter->inputs()[0];
    float* input_tensor = interpreter->typed_tensor<float>(input_idx); 
    memcpy(input_tensor, input, IN_TENSOR_SIZE * sizeof(float));

    // Invoke interpreter
    interpreter->Invoke();

    // Get output tensor
    int output_idx = interpreter->outputs()[0];
    float *output_tensor = interpreter->typed_output_tensor<float>(output_idx);

    // Convert output to image
    cv::Mat mask = postprocess(output_tensor);

    return mask;
}


float* MaskPredictor::preprocess(cv::Mat im) {
    // resize and convert
    cv::resize(im, im, cvSize(IN_SIZE, IN_SIZE));
    im.convertTo(im, CV_32F);

    // convert to float ptr
    float* input = (float*)im.data;

    // normalize
    for (int i = 0; i < OUT_TENSOR_SIZE; i++) {
        input[i] = ((input[i] / 255.0f) - 0.5f) * 2.0f; 
    }

    return input;
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
