import json
import time
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class RTDetrV2TestNode(Node):
    def __init__(self):
        super().__init__('rtdetr_v2_test_node')

        # ---------------------------------------------------------
        # 1. 파라미터 및 경로 설정
        # ---------------------------------------------------------
        self.declare_parameter('input_topic', '/camera/color/image_raw')
        self.declare_parameter('score_threshold', 0.5)
        self.declare_parameter('input_size', 640)
        self.declare_parameter('timer_period', 1.0)
        self.declare_parameter('model_dir', '')

        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.score_threshold = self.get_parameter('score_threshold').get_parameter_value().double_value
        self.input_size = int(self.get_parameter('input_size').get_parameter_value().integer_value)
        self.timer_period = float(self.get_parameter('timer_period').value)

        self.model_dir = self.resolve_model_dir(self.get_parameter('model_dir').value)
        self.onnx_path = self.model_dir / 'onnx' / 'model.onnx'
        self.config_path = self.model_dir / 'config.json'

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        if not self.config_path.exists():
            self.get_logger().warn(f"config.json not found: {self.config_path}")
        
        # ---------------------------------------------------------
        # 2. 모델 및 설정 로드
        # ---------------------------------------------------------
        self.get_logger().info(f"Loading model from {self.onnx_path}...")
        
        # ONNX Runtime 세션 생성 (Jetson이라면 CUDAProvider 우선 시도)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        except Exception as e:
            self.get_logger().warn(f"CUDA Provider failed, falling back to CPU: {e}")
            self.session = ort.InferenceSession(str(self.onnx_path), providers=['CPUExecutionProvider'])

        # 입력/출력 노드 이름 확인
        self.input_name = self.session.get_inputs()[0].name
        
        # 라벨 정보 로드 (config.json)
        self.labels = self.load_labels(self.config_path)
        
        # ---------------------------------------------------------
        # 3. ROS 통신 설정
        # ---------------------------------------------------------
        self.bridge = CvBridge()
        self.lock = Lock()
        self.latest_frame = None
        
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        # RealSense 컬러 이미지 토픽 구독
        self.sub = self.create_subscription(Image, self.input_topic, self.image_callback, qos)
        
        # 1Hz 실행을 위한 타이머 (1.0초)
        self.timer = self.create_timer(self.timer_period, self.run_inference)
        
        # 설정값
        self.input_shape = (self.input_size, self.input_size) # 모델 입력 크기

        # OpenCV 시각화 창
        self.window_name = "RT-DETR v2 Test"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.get_logger().info("RT-DETR v2 test node initialized. Running at 1Hz.")

    def load_labels(self, config_path):
        """config.json에서 id2label 정보를 읽어옵니다."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            # id2label 키가 있는지 확인
            if 'id2label' in config:
                return config['id2label']
            else:
                self.get_logger().warn("id2label not found in config.json. Using dummy labels.")
                return {str(i): f"Class {i}" for i in range(100)}
        except Exception as e:
            self.get_logger().error(f"Failed to load config.json: {e}")
            return {}

    def image_callback(self, msg):
        """이미지를 수신하면 최신 프레임만 변수에 업데이트합니다."""
        try:
            # ROS Image -> OpenCV Image (BGR)
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.latest_frame = frame
        except Exception as e:
            self.get_logger().error(f"CvBridge Error: {e}")

    def run_inference(self):
        """주기적으로 추론을 수행합니다."""
        with self.lock:
            if self.latest_frame is None:
                return
            original_image = self.latest_frame.copy()

        # 1. 전처리
        orig_h, orig_w, _ = original_image.shape
        
        # Resize & Normalize
        # preprocessor_config에 따라 0~1 사이 값으로 rescale (1/255)
        image_resized = cv2.resize(original_image, self.input_shape)
        image_data = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB) # ONNX는 보통 RGB를 예상함
        image_data = image_data.astype(np.float32) / 255.0
        
        # HWC -> CHW 변환 (3, input_size, input_size)
        image_data = np.transpose(image_data, (2, 0, 1))
        # Batch 차원 추가 (1, 3, input_size, input_size)
        input_tensor = np.expand_dims(image_data, axis=0)

        # 2. 추론 (Inference)
        start_ts = time.time()
        outputs = self.session.run(None, {self.input_name: input_tensor})
        infer_ms = (time.time() - start_ts) * 1000.0
        
        # RT-DETR v2 ONNX 출력은 보통 [logits, boxes] 순서이거나 이름으로 구분됨
        # HuggingFace export 기준:
        # output[0]: logits (Batch, Num_Queries, Num_Classes)
        # output[1]: boxes (Batch, Num_Queries, 4) -> (cx, cy, w, h) normalized
        
        logits = outputs[0][0] # 첫번째 배치만 사용
        boxes = outputs[1][0]
        
        # 3. 후처리 및 시각화
        scores = self.sigmoid(np.max(logits, axis=1)) # 각 박스의 최대 클래스 확률
        class_ids = np.argmax(logits, axis=1)         # 각 박스의 클래스 ID
        keep = scores > self.score_threshold

        for i in np.where(keep)[0]:
            score = scores[i]
            box = boxes[i]
            cx, cy, w, h = box
            
            # 정규화된 좌표를 원본 이미지 크기로 변환
            x1 = int((cx - w/2) * orig_w)
            y1 = int((cy - h/2) * orig_h)
            x2 = int((cx + w/2) * orig_w)
            y2 = int((cy + h/2) * orig_h)
            
            class_id = str(class_ids[i])
            label_name = self.labels.get(class_id, f"ID {class_id}")
            
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label_name}: {score:.2f}"
            cv2.putText(original_image, text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 4. 결과 시각화 (토픽 발행 대신 OpenCV 창 표시)
        cv2.imshow(self.window_name, original_image)
        cv2.waitKey(1)
        det_count = int(keep.sum())
        self.get_logger().info(f"RT-DETR v2 inference: {infer_ms:.1f} ms, detections: {det_count}")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def resolve_model_dir(self, override_path: str) -> Path:
        """모델 폴더 경로를 결정합니다. 파라미터 > 설치된 share > 소스 경로 순으로 확인."""
        if override_path:
            candidate = Path(override_path).expanduser().resolve()
            if candidate.exists():
                return candidate
            self.get_logger().warn(f"model_dir parameter points to missing path: {candidate}")

        try:
            share_dir = Path(get_package_share_directory('smart_pickupzone'))
            candidate = share_dir / 'model' / 'rtdetr_v2_r18vd-ONNX'
            if candidate.exists():
                return candidate
        except Exception as e:
            self.get_logger().warn(f"Failed to resolve package share directory: {e}")

        # 개발 워크스페이스에서의 기본 경로
        fallback = Path(__file__).resolve().parent / 'model' / 'rtdetr_v2_r18vd-ONNX'
        if fallback.exists():
            return fallback

        # 존재하지 않으면 마지막으로 fallback 반환 (상단에서 에러 처리)
        return fallback

    def destroy_node(self):
        cv2.destroyWindow(self.window_name)
        return super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RTDetrV2TestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
