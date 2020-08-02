//
//  MobileUNetProcessor.mm
//  rtios
//
//  Created by Igor on 31/07/2020.
//  Copyright © 2020 rusito23. All rights reserved.
//

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"
#import <opencv2/core/core.hpp>
#import "MobileUNetProcessor.h"
#import "ImageConvert.h"
#import "Utils.h"
#import "Constants.h"
#import "predict.h"
#import "transform.h"
#pragma clang pop

@interface MobileUNetProcessor ()
@property (weak, nonatomic) NSObject<MobileUnetProcessorDelegate> *delegate;
@property (nonatomic) MaskPredictor *predictor;
@property (nonatomic) Transformer *transformer;
@end

@implementation MobileUNetProcessor

#pragma mark - Threading

void dispatch_on_main(dispatch_block_t block) {
    if (block == nil) return;
    if (NSThread.isMainThread) {
        block();
    } else {
        dispatch_async(dispatch_get_main_queue(), block);
    }
}

void dispatch_on_background(dispatch_block_t block) {
    if (block == nil) return;
    dispatch_async(dispatch_queue_create(kMobileUNetProcessorSerialQueue, DISPATCH_QUEUE_SERIAL), block);
}

#pragma mark - API Impl

- (id) initWithDelegate:(NSObject<MobileUnetProcessorDelegate> *) delegate {
    if (self = [super init]) {
        // search for model path
        NSBundle *bundle = [NSBundle bundleForClass:[MobileUNetProcessor class]];
        NSString *modelPath = [bundle pathForResource:@kModelName ofType:@kModelType];
        NSLog(@"[MobileUNetProcessor] Starting predictor with model path: %@", modelPath);
        
        // init
        self.delegate = delegate;
        self.predictor = new MaskPredictor{modelPath.UTF8String, kThreshold};
        self.transformer = new Transformer();
    }
    return self;
}

- (void) captureSession:(CaptureSession *) captureSession didCaptureFrame:(CMSampleBufferRef) frame {
    // convert to opencv
    cv::Mat mat = [ImageConvert cvMatFromSampleBuffer:frame];
    
    // preprocess
    cvtColor(mat, mat, cv::COLOR_RGBA2RGB);
    
    // process
    // cv::Mat result = self.predictor->predict_mask(mat);
    
    // postprocess
    result.convertTo(result, CV_8UC3);
    cvtColor(result, result, cv::COLOR_RGB2RGBA);
    
    // convert to uiimage and set result to delegate
    UIImage* im = [ImageConvert uiImageFromCvMat:result];
    [self.delegate processor:self didProcessFrame:im];
}

#pragma mark - Helpers

const char * GetMatDepth(const cv::Mat& mat) {
    const int depth = mat.depth();
    switch (depth)
    {
        case CV_8U:  return "CV_8U";
        case CV_8S:  return "CV_8S";
        case CV_16U: return "CV_16U";
        case CV_16S: return "CV_16S";
        case CV_32S: return "CV_32S";
        case CV_32F: return "CV_32F";
        case CV_64F: return "CV_64F";
        default:
            return "Invalid depth type of matrix!";
    }
}

const char * GetMatType(const cv::Mat& mat) {
    const int mtype = mat.type();
    switch (mtype)
    {
        case CV_8UC1:  return "CV_8UC1";
        case CV_8UC2:  return "CV_8UC2";
        case CV_8UC3:  return "CV_8UC3";
        case CV_8UC4:  return "CV_8UC4";
        case CV_8SC1:  return "CV_8SC1";
        case CV_8SC2:  return "CV_8SC2";
        case CV_8SC3:  return "CV_8SC3";
        case CV_8SC4:  return "CV_8SC4";
        case CV_16UC1: return "CV_16UC1";
        case CV_16UC2: return "CV_16UC2";
        case CV_16UC3: return "CV_16UC3";
        case CV_16UC4: return "CV_16UC4";
        case CV_16SC1: return "CV_16SC1";
        case CV_16SC2: return "CV_16SC2";
        case CV_16SC3: return "CV_16SC3";
        case CV_16SC4: return "CV_16SC4";
        case CV_32SC1: return "CV_32SC1";
        case CV_32SC2: return "CV_32SC2";
        case CV_32SC3: return "CV_32SC3";
        case CV_32SC4: return "CV_32SC4";
        case CV_32FC1: return "CV_32FC1";
        case CV_32FC2: return "CV_32FC2";
        case CV_32FC3: return "CV_32FC3";
        case CV_32FC4: return "CV_32FC4";
        case CV_64FC1: return "CV_64FC1";
        case CV_64FC2: return "CV_64FC2";
        case CV_64FC3: return "CV_64FC3";
        case CV_64FC4: return "CV_64FC4";
        default:
            return "Invalid type of matrix!";
    }
}

- (void) logMat:(cv::Mat) mat withMessage:(NSString *) message {
    double min, max;
    cv::minMaxLoc(mat, &min, &max);
    NSLog(@"[MobileUNetProcessor] - %@ - channels: %d - depth: %s - type: %s - min/max: %f/%f - elemSize: %zu - total: %zu",
          message,
          mat.channels(),
          GetMatDepth(mat),
          GetMatType(mat),
          min,
          max,
          mat.elemSize(),
          mat.total());
}

@end
