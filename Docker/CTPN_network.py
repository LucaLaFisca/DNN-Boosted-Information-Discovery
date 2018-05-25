from __future__ import print_function

from PIL import Image

from redis import Redis, RedisError
import os
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from CTPN.demo import ctpn
from lib.fast_rcnn.config import cfg,cfg_from_file
from lib.networks.factory import get_network
import glob
import shutil

from flask import Flask, redirect, request, url_for, send_file
from redis import Redis
redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)
app = Flask(__name__)

cfg_from_file('CTPN/ctpn/text.yml')

# init session
sess_CTPN = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
# load network
net = get_network("VGGnet_test")
# load model
print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
saver = tf.train.Saver()

try:
    ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
    print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
    saver.restore(sess_CTPN, ckpt.model_checkpoint_path)
    print('done')
except:
    raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

def reset_CTPN():
    tf.reset_default_graph()


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
    return(compute_CTPN(img_path))

def write_output(boxes):
    file = open('boxes.txt','w')
    for box in boxes:
        file.write(str(box[0]) + ',' + str(box[1]) + ',' + str(box[2]) + ',' + str(box[3]) + ',' + str(box[4]) + ',' + str(box[5]) + ',' + str(box[6]) + ',' + str(box[7]) + ',' + str(box[8]) + ',')

    file.close()

def compute_CTPN(img_path):

    #im_name = os.path.join(cfg.DATA_DIR, filename)
    
    print(img_path)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(('Demo for {:s}'.format(img_path)))
    boxes = ctpn(sess_CTPN, net, img_path)
    print('outside CTPN_demo')
    write_output(boxes)

    return(os.path.join('output', 'CTPN.jpg'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002)


