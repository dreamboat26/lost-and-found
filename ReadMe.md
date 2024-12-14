
# Project is a framework to detect the objects and track it for fix set of frames.

### Here we have used YoloV5 as a detector and Kalman tracker as tracking algorithm.

# Requirements for GPU backend inference
1. conda create -n hackathon python=3.9.13
2. conda activate hackathon
3. conda install cudatoolkit=11.3.1 -c nvidia
4. conda install cudnn=8.2.1 -c nvidia
5. pip install -r requirements_gpu.txt

# Requirements for CPU backend inference
1. conda create -n hackathon python=3.9.13
2. conda activate hackathon
3. pip install -r requirements_cpu.txt

# Weights
- The detector model is YoloV5 which is trained on [Indian Driving Dataset](https://www.kaggle.com/datasets/manjotpahwa/indian-driving-dataset) taken from Kaggle.
- We applied some cleanup to the dataset before using.
- Weights can be downloaded from [link](https://drive.google.com/file/d/1OQWbFYv1K5lzis1G1BChJGdRzkWe6-ja/view?usp=share_link).

# Usage:
> 
    main.py [-h] [--img_path IMG_PATH] [--img_dir_path IMG_DIR_PATH] [--vid_path VID_PATH] --cls_map CLASS_MAPPING_FILE
               --model_path MODEL_PATH [--filtered_classes FILTERED_CLASSES [FILTERED_CLASSES ...]]
               [--det_iou DETECTION_IOU_THRESHOLD] [--det_score DETECTION_SCORE_THRESHOLD] [--track_iou TRACKER_IOU_THRESHOLD]       
               [--output_dir OUTPUT_DIR] [--vid_det_interval VIDEO_DETECTION_INTERVAL] [--device DEVICE] [--infer_on_grayscale]      

    Detection Tracking Framework

    optional arguments:
    -h, --help            show this help message and exit
    --img_path IMG_PATH   Image path to infer upon.
    --img_dir_path IMG_DIR_PATH
                            Directory containing images to infer upon.
    --vid_path VID_PATH   Video file to be used for inference.
    --cls_map CLASS_MAPPING_FILE
                            Class name mapping file.
    --model_path MODEL_PATH
                            Path of the trained model.
    --filtered_classes FILTERED_CLASSES [FILTERED_CLASSES ...]
                            Only show boxes of these classes.
    --det_iou DETECTION_IOU_THRESHOLD
                            IOU Threshold for NMS in the Detection operation.
    --det_score DETECTION_SCORE_THRESHOLD
                            Score Threshold for NMS in the Detection operation.
    --track_iou TRACKER_IOU_THRESHOLD
                            IOU Threshold for Tracking operation.
    --output_dir OUTPUT_DIR
                            Output directory to store results.
    --vid_det_interval VIDEO_DETECTION_INTERVAL
                            Detection interval for video.
    --device DEVICE       Execution device. Available devices are 'cpu' and 'gpu'.
    --infer_on_grayscale  Run inference on gray scale image.

# Sample commands:
1. Inference on sample image.
>
    python main.py --cls_map class_mapping_idd.txt --model_path trained_model/model.onnx --img_path "./samples/sample_idd.jpg"

2. Inference on directory of images.
>
    python main.py --cls_map class_mapping_idd.txt --model_path trained_model/model.onnx --img_dir_path "./samples"

3. Inferece on video file.
> 
    python main.py --cls_map class_mapping_idd.txt --model_path trained_model/model.onnx --vid_path "./samples/sample_vid.mp4"
