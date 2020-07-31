//
//  CaptureSession.m
//  rtios
//
//  Created by Igor on 31/07/2020.
//  Copyright © 2020 rusito23. All rights reserved.
//

#import "CaptureSession.h"
#import "UIImage+SampleBuffer.h"

@interface CaptureSession () <AVCaptureVideoDataOutputSampleBufferDelegate>
@property (weak, nonatomic) NSObject<CaptureSessionDelegate> *delegate;
@property (strong, nonatomic) AVCaptureSession *session;
@end

@implementation CaptureSession

#pragma mark - public API

- (id) initWithDelegate:(NSObject<CaptureSessionDelegate>*)delegate {
    if (self = [super init]) {
        self.delegate = delegate;
        self.session = [AVCaptureSession new];
    }
    return self;
}

- (void) startSession {
    // setup device input
    
    NSError *error;
    AVCaptureDeviceDiscoverySession *discoverySession = [AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeBuiltInWideAngleCamera]
                                                                                                               mediaType:AVMediaTypeVideo
                                                                                                                position:AVCaptureDevicePositionFront];
    AVCaptureDeviceInput *deviceInput = [AVCaptureDeviceInput deviceInputWithDevice:discoverySession.devices.firstObject error:&error];
    
    // setup video output
    
    AVCaptureVideoDataOutput *videoOutput = [AVCaptureVideoDataOutput new];
    [videoOutput setVideoSettings:@{(NSString*)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)}];
    [videoOutput setSampleBufferDelegate:self queue:dispatch_get_main_queue()];
    
    // register input / outputs
    
    if ([self.session canAddOutput:videoOutput]) {
        [self.session addOutput:videoOutput];
    }
    
    if ([self.session canAddInput:deviceInput]) {
        [self.session addInput:deviceInput];
    }
    
    // start session
    [self.session startRunning];
}

- (void) endSession {
    [self.session stopRunning];
}

#pragma mark - Sample Buffer Delegate

-(void) captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection {
    UIImage *frame = [UIImage fromSampleBuffer:sampleBuffer];
    [self.delegate captureSession:self didCaptureFrame:frame];
}

@end
