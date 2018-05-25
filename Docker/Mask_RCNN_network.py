from __future__ import print_function

from PIL import Image

from redis import Redis, RedisError
import os
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import skimage.io
import Mask_RCNN.coco as coco
import Mask_RCNN.utils as utils
import Mask_RCNN.model as modellib
import Mask_RCNN.visualize

from flask import Flask, redirect, request, url_for, send_file
from redis import Redis
redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)
app = Flask(__name__)


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


ROOT_DIR = os.getcwd()
ROOT_DIR = os.path.join(ROOT_DIR, 'Mask_RCNN')
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

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
    return(compute_Mask_RCNN(img_path))

@app.route('/score')
def main2():
    img_path = request.args.get('key1')
    return(compute_Mask_RCNN(img_path, False))


def write_output(label, coord):
    
    file = open('label.txt','w')
    for i in range(len(label)):
        file.write(label[i] + ',')
    file.close()

    file = open('coord.txt','w')
    for i in range(len(coord)):
        file.write(str(coord[i]) + ',')
    file.close()


def compute_Mask_RCNN(img_path, mask=True):
    coordinate = []
    label_name_vector = []
  
    image = skimage.io.imread(img_path)

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]

    if mask:
        Mask_RCNN.visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        img_output_path = os.path.join('output', 'Mask_RCNN.jpg')
        return img_output_path
    else:
        classes = []
        for i in range(len(r['class_ids'])):
            classes.append(class_names[r['class_ids'][i]])
        coord_1 = np.array(r['rois'])

        # y1,x1,y2,x2 --> x1,y1,x2,y2:
        coord  = np.copy(coord_1)
        coord[:,0] = coord_1[:,1] #x1
        coord[:,1] = coord_1[:,0] #y1
        coord[:,2] = coord_1[:,3] #x2
        coord[:,3] = coord_1[:,2] #y2

        coord = coord.flatten()
        write_output(classes, coord)

        #return is useless but usefull to avoid 'error' with Flask
        return img_path




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003)
