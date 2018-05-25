from __future__ import print_function

# import keras
import keras

# import keras_retinanet
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

# import miscellaneous modules
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import os
import numpy as np
import time
import glob

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

from flask import Flask, redirect, request, url_for, send_file
from redis import Redis
redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)
app = Flask(__name__)

model_path = os.path.join('keras_retinanet', 'resnet50_csv.h5')
print(model_path)

# load retinanet model
model = keras.models.load_model(model_path, custom_objects=custom_objects)
#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'Face'}



@app.route('/running')
def test():
    return 'ok'

@app.route('/compute')
def main():
    img_path = request.args.get('key1')
    return(face(img_path))

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


def write_output(label, coord):
    file = open('label.txt','w')
    for i in range(len(label)):
        file.write(label[i] + ',')
    file.close()

    file = open('coord.txt','w')
    for i in range(len(coord)):
        file.write(str(coord[i]) + ',')
    file.close()

def face(img_path):
    coordinate = []
    label_name_vector = []

    # load image
    image = read_image_bgr(img_path)
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
    
    # process image
    start = time.time()
    _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)
    
    # compute predicted labels and scores
    predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
    scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]
    
    # correct for image scale
    detections[0, :, :4] /= scale
    
    # visualize detections
    for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
        if score < 0.5:
            continue
        b = detections[0, idx, :4].astype(int)
        cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
        caption = "{:.3f}".format(score)
        cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
        cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
     
        coordinate.append(b[0])
        coordinate.append(b[1])
        coordinate.append(b[2])
        coordinate.append(b[3])
        label_name_vector.append(labels_to_names[label])

    write_output(label_name_vector, coordinate)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    img_output_path = os.path.join('output', 'retina_net_face.jpg')
    plt.imsave(img_output_path, draw)
    plt.show()
    
    return(img_output_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8008)
