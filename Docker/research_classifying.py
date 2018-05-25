from __future__ import print_function

from networks import network

from flask import Flask, redirect, request, url_for, send_file
from redis import Redis
import requests
import numpy as np
import cv2
import os
import imageio
import csv
import time

redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)
app = Flask(__name__)

@app.route('/running')
def test():
    return 'ok'

@app.route('/compute')
def main():
    img_path = request.args.get('key1')
    return(compute_classifying(img_path))

@app.route('/360')
def main2():
    img_path = request.args.get('key1')
    return(compute_score(img_path, True))

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

def get_dimension(img_path):
    img = cv2.imread(img_path)
    width = img.shape[1]
    height = img.shape[0]
    return width, height

def vector2matrix(x, size):
    N = int(len(x)/size)

    output = np.zeros((N,size))

    for i in range(len(x)):
        output[int(i/size),i%size] = x[i]

    return output



def file_is_empty(path):
    return os.stat(path).st_size==0

def get_txt(name):
    path = name + '.txt'
    if not file_is_empty(path):
        with open(path, 'r') as file:
            for line in file:
                x = line.split(",")
                del x[-1]
        return x
    else:
        return None


def ctpn_convert(x):
    y = np.zeros((x.shape[0],4))
    
    y[:,0] = x[:,0] # xmin
    y[:,1] = x[:,1] # ymin
    y[:,2] = x[:,6] # xmax
    y[:,3] = x[:,7] # ymax
    
    return y


def get_position(x, row, column):
    pos = []

    for i in range(x.shape[0]):
        
        if((x[i,2]+x[i,0])/2 < column/3):
            temp = 'left'
            if((x[i,3]+x[i,1])/2 > row/2):
                temp = temp + 'down'
            else:
                temp = temp + 'top'

        elif((x[i,2]+x[i,0])/2 > column*2/3):
            temp = 'right'
            if((x[i,3]+x[i,1])/2 > row/2):
                temp = temp + 'down'
            else:
                temp = temp + 'top'

        else:
            temp = 'center'
            if((x[i,3]+x[i,1])/2 > row/2):
                temp = temp + 'down'
            else:
                temp = temp + 'top'

        pos.append(temp)

    return pos


def assembly(classe, pos):
    # Assembly and count same boxes
    existing = []
    number = []

    for i in range(len(classe)):
        exist = False
        for index, objet in enumerate(existing):
            if (classe[i] == objet[0]):
                if(pos[i] == objet[1]): 
                    number[index] = number[index] + 1
                    exist = True

        if(exist == False):
            existing.append([classe[i], pos[i]])
            number.append(1)

    classe = []
    pos = []

    for index, objet in enumerate(existing):
        classe.append(objet[0])
        pos.append(objet[1])

    final_list = np.zeros((len(classe),3)).astype(str)

    final_list[:,0] = classe
    final_list[:,1] = pos 
    final_list[:,2] = number

    return final_list


def csv_to_text(final_list):
    text_list = ''
    for row in final_list:
        for i in range(len(row)):
            text_list = text_list + str(row[i]) + ','
        text_list = text_list[:-1]
        text_list = text_list + '\n'
    text_list = text_list[:-1]
    return text_list



def compute_classifying(img_path):
    width, height = get_dimension(img_path)

    pos = []
    classe = []

    # --- Object detection: SSD/retina_net_coco/Mask_RCNN ---
    network_to_use = 'SSD'
    payload = {'key1': img_path}
    if network_to_use == 'SSD':
        current_network = network(8001, 'SSD_network.py')
        current_network.start_network()
        r = requests.get('http://localhost:' + str(current_network.port) + '/compute', params=payload)
        print('Network used: SSD')
    elif network_to_use == 'retina_net_coco':
        current_network = network(8009, 'retina_net_coco_network.py')
        current_network.start_network()
        r = requests.get('http://localhost:' + str(current_network.port) + '/compute', params=payload)
        print('Network used: SSD')
    elif network_to_use == 'Mask_RCNN':
        current_network = network(8003, 'Mask_RCNN_network.py')
        current_network.start_network()
        r = requests.get('http://localhost:' + str(current_network.port) + '/score', params=payload)
        print('Network used: Mask_RCNN')
    else:
        print('ERROR: No network found')
        return

    label_obj = get_txt('label')
    coord_obj = get_txt('coord')

    if coord_obj and label_obj:
        coordinates_obj = vector2matrix(coord_obj, 4)
        print('coordinates obj', coordinates_obj)

        pos.extend(get_position(coordinates_obj, height, width))
        classe.extend(label_obj)

    else:
        print('no object detected')

    current_network.stop_network()

    # --- Text detection: CTPN ---
    current_network.port, current_network.name = 8002, 'CTPN_network.py'
    current_network.start_network()

    payload = {'key1': img_path}
    r = requests.get('http://localhost:' + str(current_network.port) + '/compute', params=payload)
    coord_txt = get_txt('boxes')
    
    if coord_txt:
        coordinates_txt = vector2matrix(coord_txt, 9)
        coordinates_txt = ctpn_convert(coordinates_txt)

        for i in range(coordinates_txt.shape[0]):
            classe.append('text')

        pos.extend(get_position(coordinates_txt, height, width))

    else:
        print('no text detected')
    
    current_network.stop_network()

    # --- Face detection: retina_net (faces) ---
    network_to_use = 'face_CV_network'
    if network_to_use == 'retina_net':
        current_network.port, current_network.name = 8008, 'retina_net_network.py'
        current_network.start_network()
    elif network_to_use == 'face_CV_network':
        current_network.port, current_network.name = 8006, 'face_CV_network.py'
        current_network.start_network()

    payload = {'key1': img_path}
    r = requests.get('http://localhost:' + str(current_network.port) + '/compute', params=payload)

    coord_faces = get_txt('coord')
    
    if coord_faces:
        coordinates_faces = vector2matrix(coord_faces, 4)
        print('coordinates_faces', coordinates_faces)
        
        for i in range(coordinates_faces.shape[0]):
            classe.append('face')

        pos.extend(get_position(coordinates_faces, height, width))
        

    current_network.stop_network()

    # Assembly and count same boxes

    final_list = assembly(classe, pos)


    # Convert to text file
    final_list = csv_to_text(final_list)

    return(final_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8010)
