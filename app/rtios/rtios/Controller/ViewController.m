//
//  ViewController.m
//  rtios
//
//  Created by Igor on 31/07/2020.
//  Copyright © 2020 rusito23. All rights reserved.
//

#import "ViewController.h"
#import "CaptureSession.h"

@interface ViewController () <CaptureSessionDelegate>
@property (weak, nonatomic) IBOutlet UIImageView *imageView;
@property (strong, nonatomic) CaptureSession *captureSession;
@end

@implementation ViewController

#pragma mark - Lifecycle

- (void) viewDidLoad {
    [super viewDidLoad];
    
    // setup views
    self.imageView.contentMode = UIViewContentModeScaleAspectFill;
    
    // setup capture session
    self.captureSession = [[CaptureSession alloc] initWithDelegate:self];
}

- (void) viewDidAppear:(BOOL)animated {
    [super viewDidAppear:animated];
    [self.captureSession startSession];
}

- (void) viewWillDisappear:(BOOL)animated {
    [super viewWillDisappear:animated];
    [self.captureSession endSession];
}

#pragma mark - Capture Session Delegate

- (CALayer *) rootLayer {
    return self.view.layer;
}

- (CGRect) rootFrame {
    return self.imageView.frame;
}

- (void) captureSession:(CaptureSession *)captureSession didCaptureFrame:(UIImage *)frame {
    self.imageView.image = frame;
}

@end
