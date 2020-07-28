"""
Handles the frame, applying segmentation and blur.
"""

import cv2
from predict import Predictor
from blur_util import BlurUtil
import sys
import time
import logging


class FrameProcessor:

    def __init__(self,
                 image_size,
                 frame_delay,
                 in_size,
                 out_size,
                 border_blur,
                 back_blur,
                 model):
        self.predictor = Predictor(in_size, out_size, model)
        self.blurrer = BlurUtil(image_size=image_size,
                                border_blur=border_blur,
                                back_blur=back_blur)

    def process(self, frame):
        # get mask
        start_time = time.time()
        mask = self.predictor.get_mask(frame)
        end_time = time.time()
        logging.debug(f'Mask prediction took: {end_time - start_time} s.')

        # blur with mask
        start_time = time.time()
        result = self.blurrer.blur_background(frame, mask)
        end_time = time.time()
        logging.debug(f'Mask integration + blur took'
                      ': {end_time - start_time} s.')

        return result
