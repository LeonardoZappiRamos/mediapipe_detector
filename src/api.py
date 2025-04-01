import time
from flask import Flask, Response
from video import Webcam
from media import Preprocessing


app = Flask(__name__)

processor = Preprocessing()
webcam = Webcam(preprocessor=processor)


@app.route("/start", methods=["POST"])
def start_webcam():
    """Start the webcam."""
    webcam.start()
    return "Webcam started", 200


@app.route("/stop", methods=["POST"])
def stop_webcam():
    """Stop the webcam."""
    webcam.stop()
    return "Webcam stopped", 200


@app.route("/video_feed")
def video_feed():
    """Return the current frame from the webcam as a JPEG image."""

    def generate_frame():
        webcam.start()
        while True:
            frame = webcam.get_frame()
            if frame is None:
                webcam.stop()
                break
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)

    return Response(
        generate_frame(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    # return "Error: Webcam is not running", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
