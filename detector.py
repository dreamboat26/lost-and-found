##
# @author Meet Patel <patelmeet2012@gmail.com>
# @detector.py Class to infer the detection model
# @desc Created on 2022-11-15 6:28:14 pm
# @copyright APPI SASU
##

import os
import platform
import time
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort

from utils import convert_to_3_channel_grayscale, resize_image

np.set_printoptions(suppress=True)


class Detector:
    """
    Class responsible for detection related work
    """

    def __init__(
        self,
        onnx_model_path: str,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.5,
        filtered_classes: List[str] = [],
        cls_name_to_id_mapping: Dict = {},
        infer_on_grayscale=True,
        device="cpu",
    ):
        """Constructor of Detector module

        Args:
            onnx_model_path (str): Onnx model path
            iou_threshold (float, optional): IOU Threshold for NMS. Defaults to
                0.5.
            score_threshold (float, optional): Score Threshold for NMS. Defaults
                to 0.5.
            filtered_classes (List[str], optional): Classes to be filtered from
                output. Defaults to [].
            cls_name_to_id_mapping (Dict, optional): Mapping of class name to
                class id. Defaults to {}.
            infer_on_grayscale (bool, optional): Run inference on grayscale 3
                channel image. This is required for IDD trained model.
                Defaults to True.
            device (str, optional): Device on which inference is to be executed.
                Available devices are 'cpu' and 'gpu'. Defaults to 'cpu'.

        Raises:
            FileNotFoundError: If onnx model path is not found then this will be
                raised.
        """
        if not os.path.isfile(onnx_model_path):
            raise FileNotFoundError(f"Model file is not present at : {onnx_model_path}")

        self.model_path = onnx_model_path
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        if device.lower() == "cpu":
            if platform.system() == "Windows":
                from openvino import utils

                utils.add_openvino_libs_to_path()
            exec_providers = ["OpenVINOExecutionProvider"]
            provider_options = [{"device_type": "CPU_FP32"}]
            # It is advisable to disable ort optimizations as openvino will
            # apply device specific optimizations.
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
        else:
            exec_providers = ["CUDAExecutionProvider"]
            provider_options = None

        self.sess = ort.InferenceSession(
            self.model_path,
            sess_options,
            providers=exec_providers,
            provider_options=provider_options,
        )
        assert len(self.sess.get_inputs()) == 1, "Model should have only one input."
        assert len(self.sess.get_outputs()) == 1, "Model should have only one output."

        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.input_hw = self.sess.get_inputs()[0].shape[2:4]
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.infer_on_grayscale = infer_on_grayscale
        self.return_all_classes = True if len(filtered_classes) == 0 else False
        self.filtered_classes = filtered_classes
        self.cls_name_to_id_mapping = cls_name_to_id_mapping
        self.filtered_class_ids = [
            self.cls_name_to_id_mapping[class_name]
            for class_name in self.filtered_classes
        ]

    def __preprocess_img(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float, float, float]:
        """Function to preprocess image for inference.

        Args:
            image (np.ndarray): RGB image in numpy array format.

        Returns:
            Tuple[np.ndarray, float, float, float]: Tuple of resized image
                numpy array, pad at the top, pad at the left side and scaling factor.
        """
        resized_image, pad_h, pad_w, scale = resize_image(image, self.input_hw)
        if self.infer_on_grayscale:
            resized_image = convert_to_3_channel_grayscale(resized_image)
        resized_image = np.float32(resized_image)
        resized_image = np.expand_dims(resized_image, axis=0)
        resized_image = np.transpose(resized_image, (0, 3, 1, 2))
        resized_image = resized_image / 255.0
        return resized_image, pad_h, pad_w, scale

    def __apply_nms(self, decoded_boxes: np.ndarray) -> np.ndarray:
        """Function to apply NMS over decoded boxes and get the filtered boxes
        based on iou threshold and score threshold.

        Args:
            decoded_boxes (np.ndarray): Decoded boxes coming out of Detector
                in the shape [1, 25200, 85]. It is having boxes in format
                [xc, yc, w, h, objectness, cls_probs....]

        Returns:
            np.ndarray: Filtered boxes in the shape [N, 6] where 6 values
                corresponds to [xc, yc, w, h, class_id, class_probability]
        """
        # Unstacking Bounding Box Coordinates
        bboxes = decoded_boxes[0, :, :4]
        probs = decoded_boxes[0, :, 5:] * decoded_boxes[0, :, 4:5]
        cls_ids = np.argmax(probs, axis=1)
        probs = np.max(probs, axis=1)
        filtered_idx = [probs > self.score_threshold][0]
        bboxes = bboxes[filtered_idx]
        probs = probs[filtered_idx]
        cls_ids = cls_ids[filtered_idx]

        x_min = bboxes[:, 0] - bboxes[:, 2] / 2
        y_min = bboxes[:, 1] - bboxes[:, 3] / 2
        x_max = bboxes[:, 0] + bboxes[:, 2] / 2
        y_max = bboxes[:, 1] + bboxes[:, 3] / 2

        # Sorting the pscores in descending order and keeping respective indices.
        sorted_idx = probs.argsort()[::-1]
        # Calculating areas of all bboxes.Adding 1 to the side values to avoid zero area bboxes.
        bbox_areas = (x_max - x_min + 1) * (y_max - y_min + 1)

        # list to keep filtered bboxes.
        filtered = []
        while len(sorted_idx) > 0:
            # Keeping highest pscore bbox as reference.
            rbbox_i = sorted_idx[0]
            # Appending the reference bbox index to filtered list.
            filtered.append(rbbox_i)

            # Calculating (xmin,ymin,xmax,ymax) coordinates of all bboxes w.r.t to reference bbox
            overlap_xmins = np.maximum(x_min[rbbox_i], x_min[sorted_idx[1:]])
            overlap_ymins = np.maximum(y_min[rbbox_i], y_min[sorted_idx[1:]])
            overlap_xmaxs = np.minimum(x_max[rbbox_i], x_max[sorted_idx[1:]])
            overlap_ymaxs = np.minimum(y_max[rbbox_i], y_max[sorted_idx[1:]])

            # Calculating overlap bbox widths,heights and there by areas.
            overlap_widths = np.maximum(0, (overlap_xmaxs - overlap_xmins + 1))
            overlap_heights = np.maximum(0, (overlap_ymaxs - overlap_ymins + 1))
            overlap_areas = overlap_widths * overlap_heights

            # Calculating IOUs for all bboxes except reference bbox
            ious = overlap_areas / (
                bbox_areas[rbbox_i] + bbox_areas[sorted_idx[1:]] - overlap_areas
            )

            # select indices for which IOU is greather than threshold
            delete_idx = np.where(ious > self.iou_threshold)[0] + 1
            delete_idx = np.concatenate(([0], delete_idx))

            # delete the above indices
            sorted_idx = np.delete(sorted_idx, delete_idx)

        # Return filtered bboxes
        res_bboxes = bboxes[filtered]
        res_scores = np.expand_dims(probs[filtered], axis=1)
        res_cls_ids = np.expand_dims(cls_ids[filtered], axis=1)
        return np.concatenate([res_bboxes, res_cls_ids, res_scores], axis=1)

    def __upscale_boxes(
        self, boxes: np.ndarray, pad_h: int, pad_w: int, scale: float
    ) -> np.ndarray:
        """Function to convert the detections from preprocessed image resolution
        to original image resolution.

        Args:
            boxes (np.ndarray): Final predicted boxes in [N, 6] shape.
            pad_h (int): Padding applied to top during image preprocessing.
            pad_w (int): Padding applied to left during image preprocessing.
            scale (float): Scalling applied to convert the padded image to
                model's input resolution.

        Returns:
            np.ndarray: Upscaled predicted boxes in [N, 6] shape.
        """
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        x1 -= pad_w
        x2 -= pad_w
        y1 -= pad_h
        y2 -= pad_h

        x1 *= scale
        x2 *= scale
        y1 *= scale
        y2 *= scale

        boxes[:, 0] = (x1 + x2) / 2
        boxes[:, 1] = (y1 + y2) / 2
        boxes[:, 2] = x2 - x1
        boxes[:, 3] = y2 - y1

        return boxes

    def __filter_classes(self, boxes: np.ndarray) -> np.ndarray:
        """Function to filter the predicted boxes based on classes provided.

        Args:
            boxes (np.ndarray): Predicted boxes in [N, 6] shape.

        Returns:
            np.ndarray: Filtered boxes in [N, 6] shape.
        """
        if self.return_all_classes:
            return boxes
        else:
            final_boxes = []
            for box in boxes:
                cls_id = box[-2]
                if cls_id in self.filtered_class_ids:
                    final_boxes.append(box)
            if len(final_boxes) != 0:
                final_boxes = np.stack(final_boxes, axis=0)
            return final_boxes

    def detect_boxes(self, image: np.ndarray) -> np.ndarray:
        """Function to use the given image and perform inference on given model
        and produce the final NMSed boxes.

        Args:
            image (np.ndarray): Raw RGB image in shape [H, W, 3].

        Returns:
            np.ndarray: Detected boxes in shape [N, 6] where 6 values
                corresponds to [xc, yc, w, h, class_id, class_probability]
        """
        input_batch, pad_h, pad_w, scale = self.__preprocess_img(image)
        # input_batch : [1, 3, 640, 640] : preprocessed image

        decoded_boxes = self.sess.run(
            [self.output_name], {self.input_name: input_batch}
        )[0]
        # decoded_boxes : [1, 25200, 85] : decoded boxes

        result_boxes = self.__apply_nms(decoded_boxes)
        result_boxes = self.__upscale_boxes(result_boxes, pad_h, pad_w, scale)
        result_boxes = self.__filter_classes(result_boxes)
        return result_boxes

    def benchmark(self) -> None:
        """Benchmark detector algorithm for its latency."""
        random_data = np.float32(
            np.random.randn(1, 3, self.input_hw[0], self.input_hw[1])
        )
        avg_time = []
        for _ in range(100):
            start = time.time()
            decoded_boxes = self.sess.run(
                [self.output_name], {self.input_name: random_data}
            )[0]
            delta = time.time() - start
            avg_time.append(delta)
        avg_time = np.mean(avg_time)  # in seconds
        print("==" * 30)
        print(f"Detector Avg inference time: {int(avg_time*1000)} ms")
        print(f"Detector Avg FPS: {int(1/avg_time)}")
        print("==" * 30)
