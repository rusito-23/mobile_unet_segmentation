#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"
#include "transform.h"
#include <iostream>
#pragma clang pop


cv::Mat Transformer::blur_background(cv::Mat image, cv::Mat mask) {
    // resize mask
    cv::resize(mask, mask, image.size());
    
    cv::Mat result;
    cv::Mat blured;
    
    // convert all images into 32bit floats
    result = cv::Mat(image.size(), CV_32FC3);
    mask.convertTo(mask, CV_32FC3, 1.0/255.0);
    image.convertTo(image, CV_32FC3, 1.0/255.0);
    
    // blur both original image & mask (to get softer borders)
    cv::blur(image, blured, cv::Size(91,91));
    cv::blur(mask, mask, cv::Size(31,31));
    
    cv::Mat M1, M2, M3;
    
    // replace original image in blured one using given mask
    cv::subtract(cv::Scalar::all(1.0), mask, M1);
    cv::multiply(M1, image, M2);
    cv::multiply(mask, blured, M3);
    cv::add(M2, M3, result);
    
    result.convertTo(result, CV_8UC3, 255.0);
    return result;
}


cv::Mat Transformer::replace_background(cv::Mat image, cv::Mat mask, cv::Mat background) {
    // resize
    cv::resize(mask, mask, image.size());
    cv::resize(background, background, image.size());
    
    cv::Mat result;
    
    // convert all images into 32bit floats
    result = cv::Mat(image.size(), CV_32FC3);
    mask.convertTo(mask, CV_32FC3, 1.0/255.0);
    image.convertTo(image, CV_32FC3, 1.0/255.0);
    background.convertTo(background, CV_32FC3, 1.0/255.0);
    
    // blur mask (to get softer borders)
    cv::blur(mask, mask, cv::Size(31,31));
    
    cv::Mat M1, M2, M3;
    
    // replace original image in background using given mask
    cv::subtract(cv::Scalar::all(1.0), mask, M1);
    cv::multiply(M1, image, M2);
    cv::multiply(mask, background, M3);
    cv::add(M2, M3, result);
    
    result.convertTo(result, CV_8UC3, 255.0);
    return result;
}
