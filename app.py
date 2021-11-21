import time
from datetime import datetime

from flask import Flask, render_template, Response, request, jsonify
import cv2

from mesh import FaceMeshDetector
from flask_socketio import SocketIO
from queue import LifoQueue
import threading
from fer import FER

from mocked.emotionsMock import all_mocked_emotions
from predictHateSpeech import predict_hate_speech

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # use 0 for web camera

pTime = 0
detector = FaceMeshDetector()

async_mode = None
socket = SocketIO(app, async_mode=async_mode, cors_allowed_origins="*")
q = LifoQueue()

emo_detector = FER(mtcnn=True)


@socket.on('connect')
def handle_connect():
    print('Client connected')


@socket.on('SEND_CHAT_MESSAGE')
def print_message(message):
    print("Socket ID: ", message)
    # print(message)

    payload = message['payload']
    chat_message = payload['message']
    result = predict_hate_speech(chat_message)
    #

    socket.emit('CHAT_MESSAGE_RECEIVED', {'type': 'CHAT_MESSAGE_RECEIVED', 'payload': {'authorId': payload['authorId'],
                                                                                       'message': chat_message,
                                                                                       'rejected': result != 'NEITHER',
                                                                                       'sentAt': payload['sentAt'],
                                                                                       'rejectReason': None if result == 'NEITHER' else result}})


class EmotionDetectorThread(threading.Thread):
    def __init__(self, queue, args=(), kwargs=None):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.queue = queue
        self.daemon = True
        self.receive_messages = args[0]

        self.index = 0

    def run(self):
        while True:
            val = self.queue.get()
            self.do_thing_with_message(val)
            time.sleep(1)

    def do_thing_with_message(self, message):
        if True:
            socket.emit('TYPE_DETECTED_EMOTION', [{'emotions': all_mocked_emotions[self.index]}])
            self.index += 1
            if self.index >= len(all_mocked_emotions):
                self.index = 0

        elif self.receive_messages:
            # with print_lock:
            current_frame = message['frame']
            time = message['time']
            # print('Thread', time)
            # print(current_frame)
            captured_emotions_two = emo_detector.detect_emotions(current_frame)
            print('detected', captured_emotions_two)
            socket.emit('TYPE_DETECTED_EMOTION', captured_emotions_two)


thread = EmotionDetectorThread(q, args=(1,))
thread.start()


def gen_frames():  # generate frame by frame from camera
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            current = datetime.now()
            thread.queue.put({'frame': frame, 'time': current})

            meshed_frame, faces = detector.findFaceMesh(frame)

            ret, buffer = cv2.imencode('.jpg', meshed_frame)
            meshed_frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + meshed_frame_bytes + b'\r\n')  # concat frame one by one and show result


@app.route('/video-feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict-hate-speech', methods=['POST'])
def hate_speech():
    data = request.json
    result = predict_hate_speech(data['message'])
    return jsonify({'result': result})


@app.route('/mock', methods=['POST'])
def mock():
    data = request.json

    socket.emit(data['type'],data)
    return jsonify({'result': 'OK'})


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    socket.run(app, debug=True)
