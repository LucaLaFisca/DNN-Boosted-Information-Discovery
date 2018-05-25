import sys
import os
import dlib
import glob
from skimage import io
import cv2

from flask import Flask, redirect, request, url_for, send_file
from redis import Redis
redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)
app = Flask(__name__)

predictor_path = 'dlib/examples/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

radius = 3

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/running')
def test():
    return 'ok'

@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

@app.route('/compute')
def main():
    img_path = request.args.get('key1')
    return(compute_landmark(img_path))


def compute_landmark(img_path):

    img = cv2.imread(img_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dets = detector(img, 1)

    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
        
        for i in range(shape.num_parts):
            img = cv2.circle(img,(shape.part(i).x,shape.part(i).y), radius, (0,0,255), -1)

    #dlib.hit_enter_to_continue()
    img_output_path = os.path.join('output', 'face_landmark.jpg')
    cv2.imwrite(img_output_path , img)

    return(img_output_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8005)


