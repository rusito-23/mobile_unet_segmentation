#include <iostream>
#include <zip.h>
#include <opencv2/core/core.hpp>
#include "cxxopts/cxxopts.h"
#include "predict/predict.h"
#include "transform/transform.h"

using namespace std;


//*****************************************************************************
// main
int main(int argc, char* argv[]) {

  //*************************************************************************
  // Initialize vars

  MaskPredictor *predictor;
  Transformer *transformer;

  // utils
  string title;
  cv::Mat background;
  bool has_background;

  // arguments
  const char *model_file;
  bool show_raw_mask;
  bool show_blurred_mask;
  int width;
  int height;
  string background_file;
  float thres;

  //*************************************************************************
  // Arguments

  try {
    // create argument parser
    cxxopts::Options options("RTCPP", "Realtime portrait segmentation test");
    options.add_options()
      ("model-file", ".tflite file model.", cxxopts::value<string>())
      ("show-raw-mask", "flag to show raw mask (bool)", cxxopts::value<bool>()->default_value("false"))
      ("show-blurred-mask", "flag to show the blurred mask (bool)", cxxopts::value<bool>()->default_value("false"))
      ("width", "Preview width", cxxopts::value<int>()->default_value("600"))
      ("height", "Preview height", cxxopts::value<int>()->default_value("400"))
      ("background", "Background replacement", cxxopts::value<string>()->default_value("NO"))
      ("threshold", "Segmentation threshold", cxxopts::value<float>()->default_value("0.5"))
    ;
    auto results = options.parse(argc, argv);

    // retrieve arguments
    title = results["model-file"].as<string>();
    model_file = title.c_str();
    show_raw_mask = results["show-raw-mask"].as<bool>();
    show_blurred_mask = results["show-blurred-mask"].as<bool>();
    width = results["width"].as<int>();
    height = results["height"].as<int>();
    background_file = results["background"].as<string>();
    thres = results["threshold"].as<float>();

    cout << "Model file: " << model_file << endl
         << "Show raw mask: " << show_raw_mask << endl
         << "Show blurred mask: " << show_blurred_mask << endl
         << "Width: " << width << endl
         << "Height: " << height << endl
         << "Segmentation threshold: " << thres << endl;

    // initialize utils
    predictor = new MaskPredictor(model_file, thres);
    transformer = new Transformer(cvSize(width, height), show_raw_mask, show_blurred_mask);

    // background initialization
    has_background = strcmp(background_file.c_str(), "NO") != 0;
    if (has_background) {
      cout << "Replacing background" << endl;
      background = cv::imread(background_file);
    } else {
      cout << "Bluring background" << endl;
    }

  } catch (const cxxopts::OptionException& e) {
    cout << "Error parsing options: " << e.what() << endl;
  }

  // prevent initialization errors
  if (predictor == NULL || transformer == NULL) {
    cout << "Failed to initalize predictor/transformer" << endl;
    return -1;
  }

  //*************************************************************************
  // Video capture & processing

  cv::VideoCapture cap(0);

  if (!cap.isOpened()) {
    cout << "Error opening video stream or file" << endl;
    return -1;
  }

  while(1) {

    cv::Mat im;
    // Capture frame-by-frame
    cap >> im;
    if (im.empty())
    break;

    // Get mask
    cv::Mat mask = predictor->predict_mask(im);

    // Transform
    cv::Mat output_image;

    if (has_background) {
      output_image = transformer->replace_background(im, mask, background);
    } else {
      output_image = transformer->blur_background(im, mask);
    }

    // Display the resulting image
    cv::imshow(title, output_image);

    // Press  ESC on keyboard to exit
    char c=(char)cv::waitKey(25);
    if(c==27)
    break;
  }

  // Release video capture & release windows
  cap.release();
  cv::destroyAllWindows();

  return 0;
}
