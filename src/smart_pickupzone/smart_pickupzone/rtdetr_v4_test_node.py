#!/usr/bin/env python3
from pathlib import Path
from threading import Lock
import time

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
import torch
import torchvision.transforms as T
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def resize_with_aspect_ratio(image: PILImage.Image, size: int):
    """Resize to square with padding while keeping aspect ratio."""
    original_width, original_height = image.size
    ratio = min(size / original_width, size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    image = image.resize((new_width, new_height), PILImage.BILINEAR)

    pad_w = (size - new_width) // 2
    pad_h = (size - new_height) // 2
    new_image = PILImage.new("RGB", (size, size))
    new_image.paste(image, (pad_w, pad_h))
    return new_image, ratio, pad_w, pad_h


class RTDetrV4TestNode(Node):
    def __init__(self):
        super().__init__("rtdetr_v4_test_node")

        # Parameters
        self.declare_parameter("input_topic", "/camera/color/image_raw")
        self.declare_parameter("timer_period", 1.0)
        self.declare_parameter("score_threshold", 0.5)
        self.declare_parameter("input_size", 640)
        # model_path can be a file or a directory containing RTv4-S-hgnet.onnx
        self.declare_parameter("model_path", "")

        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.timer_period = float(self.get_parameter("timer_period").value)
        self.score_threshold = (
            self.get_parameter("score_threshold").get_parameter_value().double_value
        )
        self.input_size = int(self.get_parameter("input_size").get_parameter_value().integer_value)

        # Resolve model path (param > package share > source tree)
        model_param = self.get_parameter("model_path").get_parameter_value().string_value
        self.model_path = self.resolve_model_path(model_param)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self.get_logger().info(f"Loading ONNX model: {self.model_path}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        except Exception as e:
            self.get_logger().warn(f"CUDA provider failed, using CPU. Reason: {e}")
            self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.get_logger().info(f"ONNX providers: {self.session.get_providers()}")

        self.bridge = CvBridge()
        self.lock = Lock()
        self.latest_frame = None

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Image, self.input_topic, self.image_callback, qos)

        # 1 Hz timer
        self.timer = self.create_timer(self.timer_period, self.run_inference)

        self.window_name = "RT-DETR v4 Test"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.get_logger().info("RT-DETR v4 test node initialized.")

    def resolve_model_path(self, override_path: str) -> Path:
        """Determine ONNX model path with reasonable fallbacks."""
        filename = "RTv4-S-hgnet.onnx"
        if override_path:
            candidate = Path(override_path).expanduser().resolve()
            if candidate.is_file():
                return candidate
            if candidate.is_dir():
                candidate_file = candidate / filename
                if candidate_file.exists():
                    return candidate_file
                self.get_logger().warn(f"model_path directory set but model missing: {candidate_file}")
            else:
                self.get_logger().warn(f"model_path does not exist: {candidate}")

        try:
            share_dir = Path(get_package_share_directory("smart_pickupzone"))
            candidate = share_dir / "model" / filename
            if candidate.exists():
                return candidate
        except Exception as e:
            self.get_logger().warn(f"Failed to resolve package share directory: {e}")

        # Source tree fallback
        fallback = Path(__file__).resolve().parent / "model" / filename
        if not fallback.exists():
            self.get_logger().warn(f"Fallback model path missing: {fallback}")
        return fallback

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.latest_frame = frame

    def run_inference(self):
        with self.lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()

        orig_h, orig_w = frame.shape[:2]

        pil_im = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        resized, ratio, pad_w, pad_h = resize_with_aspect_ratio(pil_im, self.input_size)
        orig_size = torch.tensor([[resized.size[1], resized.size[0]]], dtype=torch.int64)
        im_data = T.ToTensor()(resized).unsqueeze(0)

        start_ts = time.time()
        try:
            labels, boxes, scores = self.session.run(
                None,
                {"images": im_data.numpy(), "orig_target_sizes": orig_size.numpy()},
            )
            infer_ms = (time.time() - start_ts) * 1000.0
        except Exception as e:
            self.get_logger().error(f"ONNX inference failed: {e}")
            return

        labels = labels[0]
        boxes = boxes[0]
        scores = scores[0]

        keep = scores > self.score_threshold
        kept_labels = labels[keep]
        kept_boxes = boxes[keep]
        kept_scores = scores[keep]

        for lbl, bb, score in zip(kept_labels, kept_boxes, kept_scores):
            # Map back from padded square to original frame size
            x0 = int((bb[0] - pad_w) / ratio)
            y0 = int((bb[1] - pad_h) / ratio)
            x1 = int((bb[2] - pad_w) / ratio)
            y1 = int((bb[3] - pad_h) / ratio)

            x0 = max(0, min(x0, orig_w - 1))
            x1 = max(0, min(x1, orig_w - 1))
            y0 = max(0, min(y0, orig_h - 1))
            y1 = max(0, min(y1, orig_h - 1))

            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{int(lbl)}:{score:.2f}",
                (x0, max(0, y0 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

        det_count = int(keep.sum())
        self.get_logger().info(
            f"RT-DETR v4 inference: {infer_ms:.1f} ms, detections: {det_count}"
        )

    def destroy_node(self):
        cv2.destroyWindow(self.window_name)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RTDetrV4TestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
