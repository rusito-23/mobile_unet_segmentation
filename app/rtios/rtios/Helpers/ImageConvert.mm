//
//  ImageConvert.m
//  rtios
//
//  Created by Igor on 31/07/2020.
//  Copyright © 2020 rusito23. All rights reserved.
//

#import "ImageConvert.h"

#ifdef __cplusplus
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"
#import <opencv2/core/core.hpp>
#import <opencv2/imgproc/imgproc.hpp>
#pragma clang pop
#endif

@implementation ImageConvert
#ifdef __cplusplus

+ (cv::Mat) cvMatFromSampleBuffer:(CMSampleBufferRef)sampleBuffer {
    // get buffer
    CVImageBufferRef buffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    CVPixelBufferLockBaseAddress(buffer, 0);

    // get the address to the image data
    void *bufPtr = CVPixelBufferGetBaseAddressOfPlane(buffer, 0);

    // get image properties
    int width = (int)CVPixelBufferGetWidth(buffer);
    int height = (int)CVPixelBufferGetHeight(buffer);

    // create the cv mat
    cv::Mat mat;
    mat.create(height, width, CV_8UC4);
    memcpy(mat.data, bufPtr, width * height * 4);
    CVPixelBufferUnlockBaseAddress(buffer, 0);
    
    // convert to cv standard
    cv::transpose(mat, mat);
    cvtColor(mat, mat, cv::COLOR_RGBA2BGRA);

    return mat;
}

+ (UIImage *) uiImageFromCvMat:(cv::Mat)mat {
    return [ImageConvert uiImageFromCvMat:mat withGrayScale:NO];
}

+ (UIImage *) uiImageFromGrayCvMat:(cv::Mat)mat {
    return [ImageConvert uiImageFromCvMat:mat withGrayScale:YES];
}

+ (UIImage *) uiImageFromCvMat:(cv::Mat)mat withGrayScale:(BOOL) grayScale {
    // setup initial data
    NSData *data = [NSData dataWithBytes:mat.data length:mat.elemSize()*mat.total()];
    CGColorSpaceRef colorSpace = grayScale ? CGColorSpaceCreateDeviceGray() : CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);

    // create cgimage
    CGImageRef imageRef = CGImageCreate(mat.cols,
                                        mat.rows,
                                        8, 8 * mat.elemSize(),
                                        mat.step[0],
                                        colorSpace,
                                        kCGImageAlphaNoneSkipLast,
                                        provider,
                                        NULL, true,
                                        kCGRenderingIntentDefault);

    // create uiimage from cgimage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);

    return finalImage;
}

#endif
@end
