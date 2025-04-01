import cv2
import time
from media import Preprocessing


class Webcam:
    def __init__(self, preprocessor: Preprocessing, index: int = 0):
        """Initialize the WebcamManager with the specified camera index."""
        self.index = index
        self.cap = None
        self.preprocessor = preprocessor
        self.is_running = False
        self.counter = 0

    def start(self):
        """Start the webcam and begin capturing video."""
        if not self._is_running():
            self.cap = cv2.VideoCapture(self.index)

            if not self.cap.isOpened():
                print("Error: Could not open webcam.")
                return

            self.is_running = True
            # print("Webcam started.")
            # self._capture_video()

    def stop(self):
        """Stop the webcam and release the video capture object."""
        if self._is_running():
            self.is_running = False
            self.cap.release()
            self.preprocessor.release()
            cv2.destroyAllWindows()
            print("Webcam stopped.")

    def _is_running(self):
        """Check if the webcam is currently running."""
        return self.is_running

    def get_frame(self):
        """Capture a frame from the webcam and return it as a JPEG image."""
        if self._is_running():
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame.")
                return None

            frame = cv2.flip(frame, 1)
            start_time = time.time()
            processed_frame = self.preprocessor.run_detection(frame, start_time)

            if processed_frame is not None:
                # Encode the processed_frame as JPEG
                _, jpeg = cv2.imencode(".jpg", processed_frame)
            else:
                # Encode the frame as JPEG
                print("Frame unprocessed")
                _, jpeg = cv2.imencode(".jpg", frame)
            return jpeg.tobytes()
        return None
