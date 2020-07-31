//
//  UIImage+SampleBuffer.m
//  rtios
//
//  Created by Igor on 31/07/2020.
//  Copyright © 2020 rusito23. All rights reserved.
//

#import "UIImage+SampleBuffer.h"

@implementation UIImage (SampleBuffer)

+ (id)fromSampleBuffer:(CMSampleBufferRef) sampleBuffer {
    // Get Image Buffer
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
    
    // Metadata
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
    size_t width = CVPixelBufferGetWidth(imageBuffer);
    size_t height = CVPixelBufferGetHeight(imageBuffer);
    
    
    // Get the number of bytes per row for the pixel buffer
    u_int8_t *baseAddress = (u_int8_t *)malloc(bytesPerRow*height);
    memcpy(baseAddress, CVPixelBufferGetBaseAddress(imageBuffer), bytesPerRow * height);
    
    // size_t bufferSize = CVPixelBufferGetDataSize(imageBuffer);
    
    // Create a device-dependent RGB color space
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    
    // Create a bitmap graphics context with the sample buffer data
    
    //The context draws into a bitmap which is `width'
    //  pixels wide and `height' pixels high. The number of components for each
    //      pixel is specified by `space'
    CGContextRef context = CGBitmapContextCreate(baseAddress,
                                                 width, height,
                                                 8,
                                                 bytesPerRow,
                                                 colorSpace,
                                                 kCGBitmapByteOrder32Little | kCGImageAlphaNoneSkipFirst);
    
    // Create a Quartz image from the pixel data in the bitmap graphics context
    CGImageRef quartzImage = CGBitmapContextCreateImage(context);
    
    // Unlock the pixel buffer
    CVPixelBufferUnlockBaseAddress(imageBuffer,0);
    
    // Free up the context and color space
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    
    // Create an image object from the Quartz image
    UIImage *image = [UIImage imageWithCGImage:quartzImage scale:1.0 orientation:UIImageOrientationRight];
    
    free(baseAddress);
    // Release the Quartz image
    CGImageRelease(quartzImage);
    
    return image;
}

@end
