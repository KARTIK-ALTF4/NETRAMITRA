# OpenCV DNN Face Detection Models

Download these files and place in the project directory:

## 1. deploy.prototxt
URL: https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

## 2. res10_300x300_ssd_iter_140000_fp16.caffemodel
URL: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel

## Auto-download script:

```bash
# Download deploy.prototxt
curl -o deploy.prototxt https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

# Download model
curl -o res10_300x300_ssd_iter_140000_fp16.caffemodel https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel
```

Or the script will automatically use Haar Cascade (no downloads needed!)
