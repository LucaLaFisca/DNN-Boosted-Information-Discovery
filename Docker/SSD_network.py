from __future__ import print_function

import os
import tensorflow as tf
import sys
import numpy as np
from SSD.ssd import SSD300
from SSD.ssd_utils import BBoxUtility

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import image
from scipy.ndimage import imread
from keras.applications.imagenet_utils import preprocess_input

from flask import Flask, redirect, request, url_for, send_file
from redis import Redis
redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)
app = Flask(__name__)

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'
np.set_printoptions(suppress=True)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))

voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']

NUM_CLASSES = len(voc_classes) + 1

input_shape=(300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('SSD/weights_SSD300.hdf5', by_name=True)

bbox_util = BBoxUtility(NUM_CLASSES)

@app.route('/running')
def test():
    return 'ok'

@app.route('/compute')
def main():
    img_path = request.args.get('key1')
    return(compute_SSD(img_path))

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

def compute_SSD(img_path):
    inputs = []
    images = []
    coordinate = []
    label_name_vector = []
    
    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img)
    images.append(imread(img_path))
    inputs.append(img.copy())
    
    inputs = preprocess_input(np.array(inputs))
    
    preds = model.predict(inputs, batch_size=1, verbose=1)
    
    results = bbox_util.detection_out(preds)
    
    for i, img in enumerate(images):
        # Parse the outputs.
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]
        
        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
        
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
        
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        
        fig, ax = plt.subplots(1)#, figsize=(16, 16))
        ax.imshow(img / 255.)
        ax.axis('off')
        ax.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            
            coordinate.append(xmin)
            coordinate.append(ymin)
            coordinate.append(xmax)
            coordinate.append(ymax)
            label = int(top_label_indices[i])
            label_name = voc_classes[label - 1]
            label_name_vector.append(label_name)
            
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = voc_classes[label - 1]
            display_txt = '{:0.2f}, {}'.format(score, label_name)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label]
            ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            ax.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
            #patches.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2)
            #plt.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
    
    
        write_output(label_name_vector, coordinate)

        img_output_path = os.path.join('output', 'SSD.jpg')
    
        fig.subplots_adjust(bottom = 0)
        fig.subplots_adjust(top = 1)
        fig.subplots_adjust(right = 1)
        fig.subplots_adjust(left = 0)
        fig.savefig(img_output_path,bbox_inches='tight',transparent=True, pad_inches=0)

        #plt.savefig(img_output_path)
        return(img_output_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
