import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import cv2
import numpy as np
import threading
import time
import sys

from ultralytics import YOLO

class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.model = YOLO("yolov8s.pt")
        self.get_logger().info("YOLO model loaded. Ready for The Great Object Hunt.")

        self.bridge = CvBridge()
        
        self.image_sub = self.create_subscription(Image, 'camera/image', self.image_callback, 1)
        self.depth_sub = self.create_subscription(Image, 'camera/depth_image', self.depth_callback, 1)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.latest_frame = None
        self.latest_depth = None
        self.frame_lock = threading.Lock()
        self.running = True

        self.target_class = None
        self.mission_state = "IDLE" 
        self.current_distance = -1.0
        self.stopping_distance = 1.0 
        
        self.spin_thread = threading.Thread(target=self.spin_thread_func, daemon=True)
        self.spin_thread.start()

        self.input_thread = threading.Thread(target=self.target_input_thread, daemon=True)
        self.input_thread.start()

        self.prev_time = time.time()

    def spin_thread_func(self):
        while rclpy.ok() and self.running:
            rclpy.spin_once(self, timeout_sec=0.05)

    def target_input_thread(self):
        time.sleep(2)
        while self.running:
            if self.mission_state in ["IDLE", "COMPLETED"]:
                print("\n" + "="*40)
                print("HOME ASSISTANT READY")
                print("="*40)
                sys.stdout.flush()
                
                target = input("Enter target object: ").strip().lower()
                
                if target and self.running:
                    self.target_class = target
                    self.mission_state = "SEARCHING"
                    print(f"Searching for: {self.target_class}")
            time.sleep(0.5)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        with self.frame_lock:
            self.latest_frame = frame

    def depth_callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        with self.frame_lock:
            self.latest_depth = depth

    def stop(self):
        self.running = False
        if self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1)

    def display_image(self):
        cv2.namedWindow("The Great Object Hunt", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("The Great Object Hunt", 1200, 700)

        while rclpy.ok() and self.running:
            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
                depth = None if self.latest_depth is None else self.latest_depth.copy()

            if frame is not None:
                result = self.process_mission(frame, depth)
                cv2.imshow("The Great Object Hunt", result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cv2.destroyAllWindows()

    def process_mission(self, frame, depth_frame):
        CONF_THRESHOLD = 0.35
        results = self.model(frame, conf=CONF_THRESHOLD, imgsz=640, verbose=False)

        detections = []
        target_box = None
        highest_conf = 0.0

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]
                detections.append(f"{class_name} ({confidence:.2f})")
                
                if self.target_class and class_name.lower() == self.target_class:
                    if confidence > highest_conf:
                        highest_conf = confidence
                        target_box = (x1, y1, x2, y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, f"TARGET {confidence:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
                    cv2.putText(frame, class_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0

        if self.mission_state == "SEARCHING":
            if target_box is None:
                msg.angular.z = 0.5 
            else:
                print("Target Found!")
                self.mission_state = "TRACKING"

        if self.mission_state == "TRACKING":
            if target_box is not None:
                x1, y1, x2, y2 = target_box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if depth_frame is not None:
                    region = depth_frame[max(0, cy-2):min(depth_frame.shape[0], cy+3),
                                         max(0, cx-2):min(depth_frame.shape[1], cx+3)]
                    valid_depths = region[(region > 0) & (~np.isnan(region)) & (~np.isinf(region))]
                    if len(valid_depths) > 0:
                        self.current_distance = float(np.median(valid_depths))
                
                if 0 < self.current_distance <= self.stopping_distance:
                    self.mission_state = "COMPLETED"
                    print("\nMission Completed\nTarget Reached Successfully")
                else:
                    img_center_x = frame.shape[1] / 2
                    error_x = img_center_x - cx
                    msg.angular.z = float(error_x * 0.002) 
                    msg.linear.x = 0.25 
            else:
                self.mission_state = "SEARCHING" 

        if self.mission_state in ["COMPLETED", "IDLE"]:
            msg.linear.x = 0.0
            msg.angular.z = 0.0

        self.cmd_pub.publish(msg)

        current_time = time.time()
        fps = 1.0 / max(current_time - self.prev_time, 1e-6)
        self.prev_time = current_time
        
        dashboard_width = 350
        dashboard = np.zeros((frame.shape[0], dashboard_width, 3), dtype=np.uint8)

        cv2.putText(dashboard, "MISSION DASHBOARD", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.putText(dashboard, f"TARGET: {self.target_class or 'None'}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        status_color = (200, 200, 200)
        if self.mission_state == "SEARCHING": status_color = (0, 165, 255) 
        if self.mission_state == "TRACKING": status_color = (0, 255, 255) 
        if self.mission_state == "COMPLETED": status_color = (0, 255, 0) 
        
        cv2.putText(dashboard, f"MODE: {self.mission_state}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        if self.current_distance > 0 and self.mission_state in ["TRACKING", "COMPLETED"]:
            cv2.putText(dashboard, f"DIST: {self.current_distance:.2f} m", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(dashboard, "-"*25, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        cv2.putText(dashboard, f"FPS: {fps:.1f}", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        y = 270
        for det in detections[:15]:
            cv2.putText(dashboard, det, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            y += 25

        combined = np.hstack((frame, dashboard))
        return combined

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        node.display_image()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()