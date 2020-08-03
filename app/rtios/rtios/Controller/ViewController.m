//
//  ViewController.m
//  rtios
//
//  Created by Igor on 31/07/2020.
//  Copyright © 2020 rusito23. All rights reserved.
//

#import "ViewController.h"
#import "CaptureSession.h"
#import "MobileUNetProcessor.h"

@interface ViewController () <MobileUnetProcessorDelegate>
@property (weak, nonatomic) IBOutlet UIImageView *imageView;
@property (weak, nonatomic) IBOutlet UIButton *modeButton;
@property (strong, nonatomic) CaptureSession *captureSession;
@property (strong, nonatomic) MobileUNetProcessor *processor;
@end

@implementation ViewController

#pragma mark - Lifecycle

- (void) viewDidLoad {
    [super viewDidLoad];
    
    // setup views
    self.imageView.contentMode = UIViewContentModeScaleAspectFill;
    self.modeButton.layer.borderWidth = 2.0;
    self.modeButton.layer.borderColor = UIColor.blackColor.CGColor;
    self.modeButton.layer.cornerRadius = 10.0;
    
    // setup capture session
    self.processor = [[MobileUNetProcessor alloc] initWithDelegate:self];
    self.captureSession = [[CaptureSession alloc] initWithDelegate:self.processor];
    
    // setup mode button
    [self.modeButton addTarget:self action:@selector(onModeButtonPressed) forControlEvents:UIControlEventTouchUpInside];
    [self updateModeButton];
}

- (void) viewDidAppear:(BOOL)animated {
    [super viewDidAppear:animated];
    [self.captureSession startSession];
}

- (void) viewWillDisappear:(BOOL)animated {
    [super viewWillDisappear:animated];
    [self.captureSession endSession];
}

#pragma mark - Mode Button Handling

- (void) updateModeButton {
    switch (self.processor.mode) {
        case MUReplaceBackgroundMode: [self.modeButton setTitle:@"Background Blur" forState:UIControlStateNormal];
        case MUBackgroundBlurMode: [self.modeButton setTitle:@"Background Replacement" forState:UIControlStateNormal];
    }
}

- (void) onModeButtonPressed {
    switch (self.processor.mode) {
        case MUReplaceBackgroundMode:
            self.processor.mode = MUBackgroundBlurMode;
            break;
        case MUBackgroundBlurMode:
            self.processor.mode = MUReplaceBackgroundMode;
            break;
    }
    [self updateModeButton];
}

#pragma mark - Processor Delegate

- (void) processor:(MobileUNetProcessor *) processor didProcessFrame:(UIImage *)frame {
    self.imageView.image = frame;
}

#pragma mark - Prevent rotation

- (BOOL) shouldAutorotate {
    return NO;
}

- (UIInterfaceOrientation) preferredInterfaceOrientationForPresentation {
    return UIInterfaceOrientationPortrait;
}

@end
