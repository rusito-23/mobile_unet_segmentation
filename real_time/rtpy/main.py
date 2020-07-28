"""
Run the background segmentation live test.
"""

import cv2
import argparse
from frame_processor import FrameProcessor
from log import LoggingConfig


class BackgroundBlurTest:

    def __init__(self,
                 image_size,
                 frame_delay,
                 in_size,
                 out_size,
                 border_blur,
                 back_blur,
                 model):

        self.processor = FrameProcessor(
                                image_size=image_size,
                                frame_delay=frame_delay,
                                in_size=in_size,
                                out_size=out_size,
                                border_blur=border_blur,
                                back_blur=back_blur,
                                model=model)
        self.capture = cv2.VideoCapture(0)
        self.model = model.split('/')[-1]

    def capture_video(self):
        while True:
            ret, frame = self.capture.read()
            output = self.processor.process(frame)
            cv2.imshow(f'RT PY - {self.model}', output)
            if cv2.waitKey(1) == 27:
                break

        self.capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--width', type=int, default=700)
    parser.add_argument('-he', '--height', type=int, default=600)
    parser.add_argument('-f', '--frame_delay', type=int, default=1)
    parser.add_argument('-i', '--input_size', type=int, default=224)
    parser.add_argument('-o', '--output_size', type=int, default=112)
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-bb', '--border_blur', type=int, default=31)
    parser.add_argument('-b', '--blur', type=int, default=31)
    parser.add_argument('-l', '--level', default='info', dest='logger_level',
                        choices=LoggingConfig.get_level_config())
    args = parser.parse_args()

    LoggingConfig(args.logger_level)

    test = BackgroundBlurTest(
        image_size=(args.width, args.height),
        frame_delay=args.frame_delay,
        in_size=args.input_size,
        out_size=args.output_size,
        border_blur=args.border_blur,
        back_blur=args.blur,
        model=args.model)

    test.capture_video()
