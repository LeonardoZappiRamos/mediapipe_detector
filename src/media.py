import cv2
import time
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
DetectionResult = mp.tasks.components.containers.DetectionResult
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# MODEL_PATH = r".\src\model\efficientdet_lite0.tflite"
# MODEL_PATH = r".\src\model\ppe_model_v0.tflite"
MODEL_PATH = r".\src\model\cpee.tflite"


class Preprocessing:
    def __init__(self):
        self.MARGIN = 20  # pixels
        self.ROW_SIZE = 24  # pixels
        self.FONT_SIZE = 1
        self.FONT_THICKNESS = 1
        self.TEXT_COLOR = (255, 0, 0)  # red
        self.WIDTH = 1216
        self.HEIGHT = 800
        self.fps_avg_frame_count = 30
        self.model_path = MODEL_PATH
        self.threshold = 0.5
        self.counter = 0
        self.fps = 0
        self.start_time = None
        self.detector = None
        self.detection_result_list = []

    def visualize(self, image, detection_result) -> np.ndarray:
        """Draws bounding boxes on the input image and return it.
        Args:
            image: The input RGB image.
            detection_result: The list of all "Detection" entities to be visualize.
        Returns:
            Image with bounding boxes.
        """
        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, self.TEXT_COLOR, 3)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = category_name + " (" + str(probability) + ")"
            text_location = (
                self.MARGIN + bbox.origin_x,
                self.MARGIN + self.ROW_SIZE + bbox.origin_y,
            )
            cv2.putText(
                image,
                result_text,
                text_location,
                cv2.FONT_HERSHEY_PLAIN,
                self.FONT_SIZE,
                self.TEXT_COLOR,
                self.FONT_THICKNESS,
            )

        return image

    def start(self):
        # Initialize the object detection model
        self.base_options = BaseOptions(model_asset_path=self.model_path)
        self.options_live = ObjectDetectorOptions(
            base_options=self.base_options,
            running_mode=VisionRunningMode.LIVE_STREAM,
            score_threshold=self.threshold,
            result_callback=self.visualize_callback,
        )
        return self

    def visualize_callback(
        self,
        result,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        print("Callback called with result:", result)
        # print("Detected categories:", [d.category_name for d in result.detections])
        result.timestamp_ms = timestamp_ms
        self.detection_result_list.append(result)

    def run_detection(self, image, start_time):
        """Run the detection"""
        self.start_time = start_time
        self.start()

        if image is None or image.size == 0:
            print("Invalid image received for detection.")
            return None

        try:
            with ObjectDetector.create_from_options(self.options_live) as detection:
                self.detector = detection
                self.counter += 1

                # Convert the image from BGR to RGB as required by the TFLite model.
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(src=image, dsize=(1216, 800))
                image_np = np.array(image)
                mp_image = mp.Image(mp.ImageFormat.SRGB, image_np)
                current_frame = np.copy(mp_image.numpy_view()).copy()

                detection.detect_async(mp_image, self.counter)

                # Calculate the FPS
                if self.counter % self.fps_avg_frame_count == 0:
                    end_time = time.time()
                    self.fps = self.fps_avg_frame_count / (end_time - self.start_time)
                    self.start_time = time.time()

                # Show the FPS
                fps_text = "FPS = {:.1f}".format(self.fps)
                text_location = (self.MARGIN, self.ROW_SIZE)
                cv2.putText(
                    current_frame,
                    fps_text,
                    text_location,
                    cv2.FONT_HERSHEY_PLAIN,
                    self.FONT_SIZE,
                    self.TEXT_COLOR,
                    self.FONT_THICKNESS,
                )

                # Process the detection result.
                # print("Detection results:", self.detection_result_list)
                if self.detection_result_list:
                    annotated_image = self.visualize(
                        current_frame, self.detection_result_list[0]
                    )
                    self.detection_result_list.clear()
                    return annotated_image
                else:
                    return current_frame
        except Exception as e:
            print(f"Error during detection: {e}")
            if e.__cause__:
                print(f"Cause: {e.__cause__}")
            return None

    def release(self):
        self.detector.close()
