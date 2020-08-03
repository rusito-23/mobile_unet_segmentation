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
@property BOOL processing;
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
    if (self.processing) return;
    self.processing = YES;
    
    // convert to opencv
    cv::Mat target = [ImageConvert cvMatFromSampleBuffer:frame];
    cvtColor(target, target, cv::COLOR_RGBA2BGR);
    
    __weak typeof(self) weakSelf = self;
    dispatch_on_background(^{
        __strong typeof(weakSelf) self = weakSelf;
        
        // process
        cv::Mat mask = self.predictor->predict_mask(target);
        cv::Mat result = self.transformer->blur_background(target, mask);
        cvtColor(result, result, cv::COLOR_BGR2RGBA);
        
        // display on main thread
        dispatch_on_main(^{
            self.processing = NO;
            // convert to uiimage and set result to delegate
            UIImage* im = [ImageConvert uiImageFromCvMat:result];
            [self.delegate processor:self didProcessFrame:im];
        });
    });
}

@end
