##
# @author Meet Patel <patelmeet2012@gmail.com>
# @main.py Entry point python file to detect and track objects.
# @desc Created on 2022-11-15 6:29:26 pm
# @copyright APPI SASU
##

import argparse
import glob
import os
import shutil

import cv2

from detector import Detector
from tracker import Tracker
from utils import get_class_mapping_dict, plot_boxes, single_image_inference

parser = argparse.ArgumentParser(description="Detection Tracking Framework")
parser.add_argument(
    "--img_path", required=False, dest="img_path", help="Image path to infer upon."
)
parser.add_argument(
    "--img_dir_path",
    required=False,
    dest="img_dir_path",
    help="Directory containing images to infer upon.",
)
parser.add_argument(
    "--vid_path",
    required=False,
    dest="vid_path",
    help="Video file to be used for inference.",
)
parser.add_argument(
    "--cls_map",
    required=True,
    dest="class_mapping_file",
    help="Class name mapping file.",
)
parser.add_argument(
    "--model_path", required=True, dest="model_path", help="Path of the trained model."
)
parser.add_argument(
    "--filtered_classes",
    default=[],
    type=str,
    dest="filtered_classes",
    nargs="+",
    help="Only show boxes of these classes.",
)
parser.add_argument(
    "--det_iou",
    default=0.65,
    type=float,
    dest="detection_iou_threshold",
    help="IOU Threshold for NMS in the Detection operation.",
)
parser.add_argument(
    "--det_score",
    default=0.5,
    type=float,
    dest="detection_score_threshold",
    help="Score Threshold for NMS in the Detection operation.",
)
parser.add_argument(
    "--track_iou",
    default=0.5,
    type=float,
    dest="tracker_iou_threshold",
    help="IOU Threshold for Tracking operation.",
)
parser.add_argument(
    "--output_dir",
    default="./results",
    type=str,
    dest="output_dir",
    help="Output directory to store results.",
)
parser.add_argument(
    "--vid_det_interval",
    default=10,
    type=float,
    dest="video_detection_interval",
    help="Detection interval for video.",
)
parser.add_argument(
    "--device",
    default="cpu",
    type=str,
    dest="device",
    help="Execution device. Available devices are 'cpu' and 'gpu'.",
)
parser.add_argument(
    "--infer_on_grayscale",
    action="store_true",
    dest="infer_on_grayscale",
    help="Run inference on gray scale image.",
)

args = parser.parse_args()

if len(glob.glob(args.output_dir + "/*")) > 0:
    print("output_dir is not empty. Deleting all contents.")
    shutil.rmtree(args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)

if args.img_path is None and args.img_dir_path is None and args.vid_path is None:
    print("Please provide either --img_path, --img_dir_path or --vid_path arguments.")
    exit()

if (
    sum(
        [
            args.img_path is not None,
            args.img_dir_path is not None,
            args.vid_path is not None,
        ]
    )
    > 1
):
    print(
        "Please provide any one from --img_path, --img_dir_path or --vid_path arguments."
    )
    exit()

if not os.path.isfile(args.class_mapping_file):
    print(f"Class mapping file not found at : {args.class_mapping_file}.")
    exit()
else:
    class_mapping_dict = get_class_mapping_dict(args.class_mapping_file)

if not os.path.isfile(args.model_path):
    print(f"Model file not found at : {args.model_path}.")
    exit()
else:
    detector_model = Detector(
        args.model_path,
        args.detection_iou_threshold,
        args.detection_score_threshold,
        args.filtered_classes,
        class_mapping_dict,
        args.infer_on_grayscale,
        args.device,
    )
    detector_model.benchmark()
    tracker = Tracker(iou_threshold=args.tracker_iou_threshold)

if args.img_path is not None and not os.path.isfile(args.img_path):
    print(f"Image file not found at : {args.img_path}")
    exit()
elif args.img_path is not None:
    single_image_inference(detector_model, args.img_path, args.output_dir)
    exit()


if args.img_dir_path is not None and not os.path.isdir(args.img_dir_path):
    print(f"Image directory not found at : {args.img_dir_path}")
    exit()
elif args.img_dir_path is not None:
    img_paths = glob.glob(args.img_dir_path + "/*.jpg")
    for img_path in img_paths:
        single_image_inference(detector_model, img_path, args.output_dir)
    exit()

if args.vid_path is not None and not os.path.isfile(args.vid_path):
    print(f"Video file not found at : {args.vid_path}")
    exit()
elif args.vid_path is not None:
    video = cv2.VideoCapture(args.vid_path)
    frame_w = int(video.get(3))
    frame_h = int(video.get(4))
    fps = int(video.get(5))
    output_res_path = os.path.join(
        args.output_dir,
        os.path.splitext(os.path.basename(args.vid_path))[0] + "_result.mp4",
    )
    writer = cv2.VideoWriter(
        output_res_path, cv2.VideoWriter_fourcc(*"MP4V"), fps, (frame_w, frame_h)
    )

    ctr = 0
    while True:
        ok, frame = video.read()
        if not ok:
            break

        if ctr % args.video_detection_interval == 0:
            res = detector_model.detect_boxes(frame)
            idx_mapping = tracker.initialize_tracker(frame, res)
        else:
            idx_mapping = tracker.get_predictions(frame)

        res_frame = plot_boxes(frame, idx_mapping, verbose=False)
        writer.write(res_frame)
        res_frame = cv2.resize(res_frame, (640, 480))
        cv2.imshow("res_frame", res_frame)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break

        ctr += 1

    cv2.destroyAllWindows()
    writer.release()
