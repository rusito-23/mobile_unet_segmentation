"""
Blur the background using the mask.
"""

import cv2
import numpy as np
import logging
import time


class BlurUtil:

    def __init__(self, image_size, back_blur, border_blur):
        self.back_blur = (back_blur, back_blur)
        self.border_blur = (border_blur, border_blur)
        self.image_size = image_size

    def blur_background(self, image, mask):
        start_time = time.time()

        # resize both mask & image
        foreground = cv2.resize(image, self.image_size, cv2.INTER_NEAREST)
        mask = cv2.resize(mask, self.image_size)

        # blur image
        blurredImage = cv2.blur(foreground, self.back_blur)

        # convert to float
        foreground = foreground.astype(float)
        blurredImage = blurredImage.astype(float)

        # blur mask to get softer borders
        th, alpha = cv2.threshold(np.array(mask), 0, 255, cv2.THRESH_BINARY)
        alpha = cv2.blur(alpha, self.border_blur)
        alpha = alpha.astype(float)/255

        # join mask & image
        foreground = cv2.multiply(alpha, foreground)
        background = cv2.multiply(1.0 - alpha, blurredImage)
        out_image = cv2.add(foreground, background)
        out_image = out_image / 255

        end_time = time.time()
        logging.debug(f'Blur took: {end_time - start_time} s.')

        return out_image
