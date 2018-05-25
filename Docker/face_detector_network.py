from __future__ import print_function

import dlib
from skimage import io
import cv2
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import os
from flask import Flask, redirect, request, url_for, send_file
from redis import Redis
redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)
app = Flask(__name__)

default_width = 400
cnn_face_detector = dlib.cnn_face_detection_model_v1('dlib/examples/mmod_human_face_detector.dat')

@app.route('/running')
def test():
    return 'ok'

@app.route('/compute')
def main():
    img_path = request.args.get('key1')
    return(compute_face_detector(img_path))

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

def compute_face_detector(img_path):

    img = cv2.imread(img_path)
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if max(img.shape)>default_width:
        rescale_factor = default_width/max(img.shape)
        height = int(img.shape[0]*rescale_factor)
        width = int(img.shape[1]*rescale_factor)
        img = cv2.resize(img, (width, height))

    dets = cnn_face_detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

        top_left = (d.rect.left(), d.rect.top())
        bottom_right = (d.rect.right(),d.rect.bottom())
        color = (0,255,0)
        img = cv2.rectangle(img, top_left, bottom_right, color, 1)

    img_output_path = os.path.join('output', 'face_detector.jpg')
    cv2.imwrite(img_output_path , img)
    #fig.savefig(img_path)
    return(img_output_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8004)

