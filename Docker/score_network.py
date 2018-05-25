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
    return(compute_score(img_path))

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

def vector2matrix(x, size):
    N = int(len(x)/size)

    output = np.zeros((N,size))

    for i in range(len(x)):
        output[int(i/size),i%size] = x[i]

    return output

def area_boxes(x, img_path):
    area = []
    img = cv2.imread(img_path)
    base_area = img.shape[0]*img.shape[1]
    
    for i in range(x.shape[0]):
        area.append((x[i,2]-x[i,0])*(x[i,3]-x[i,1])/base_area)
    return area

def scale_score(area, class_type):
    if class_type == 'text':
        score = 0.4*area**(1/6)
    elif class_type == 'face':
        if area <0.05:
            score = 0.3
        else:
            score = 0.5
    else:
        if area < 0.45:
            score = area*2
        else:
            score = 0.9
    return score

def get_score_area(coord, img_path, class_type):
    area = area_boxes(coord, img_path)
    score = []
    for i in range(len(area)):
        score.append(scale_score(area[i], class_type))
    return score

def get_dimension(img_path):
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    return height, width

def get_score_position_360(coord, img_path):
    print('initial img :', get_dimension(img_path))
    height, width = get_dimension(img_path)

    matrix = cv2.imread('map.jpg')
    matrix = cv2.cvtColor(matrix, cv2.COLOR_RGB2GRAY)

    matrix = cv2.resize(matrix, (width, height))
    matrix = np.array(matrix)/255.

    score = []
    for i in range(len(coord[:,0])):
        sub_matrix = matrix[int(coord[i,1]):int(coord[i,3]+1),int(coord[i,0]):int(coord[i,2]+1)]
        score.append(np.max(sub_matrix))
    score = np.array(score)

    score = score/np.max(score)

    return score

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def get_score_position(coord, img_path):
    height, width = get_dimension(img_path)

    mu_x, sigma_x = width/2, width/5 #8
    mu_y, sigma_y = height/2, height/5 #9

    x = np.tile(gaussian(np.linspace(0, width-1, width), mu_x, sigma_x), (height, 1))

    y = np.tile(gaussian(np.linspace(0, height-1, height), mu_y, sigma_y), (width, 1))
    y = np.transpose(y)
    
    matrix = np.multiply(y, x)
    matrix = matrix*255/np.max(matrix)
    
    score = []
    for i in range(len(coord[:,0])):
        sub_matrix = matrix[int(coord[i,1]):int(coord[i,3]+1), int(coord[i,0]):int(coord[i,2]+1),]
        score.append(np.max(sub_matrix))

    score = np.array(score)
    score = score/np.max(score)

    return score

def get_score_classe(label):
    classes_DB = []
    score_DB = []
    score = []
    with open(os.path.join('CSV', 'label_score.csv'), 'rt', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            classes_DB.append(row[0])
            score_DB.append(row[1])

    for i in range(len(label)):
        index = classes_DB.index(label[i].lower())
        score.append(float(score_DB[index]))

    return(score)

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

def attention_map(boxes,score,prev_map):
    start = time.time()

    width, height = prev_map.shape[1], prev_map.shape[0]
    object_map = prev_map
    matrix = np.zeros((height, width))

    for i in range(boxes.shape[0]):
        a=int(boxes[i,0])
        b=int(boxes[i,2])
        c=int(boxes[i,1])
        d=int(boxes[i,3])
        
        mu_x = (a+b)/2
        sigma_x = (b-a)/4
        
        start_1 = time.time()
        smooth_x = np.tile(gaussian(np.linspace(0, width-1, width), mu_x, sigma_x), (height, 1))
        end_1 = time.time()
        mu_y = (c+d)/2
        sigma_y = (d-c)/4
        
        smooth_y = np.tile(gaussian(np.linspace(0, height-1, height), mu_y, sigma_y), (width, 1))
        smooth_y = np.transpose(smooth_y)

        matrix = np.multiply(smooth_y, smooth_x)
        object_map = np.maximum(score[i] * matrix, object_map)

    end = time.time()
    print('total time elapsed [s]: ', end - start)

    return object_map

def get_basic_map(height, width, path=os.path.join('basic_map', 'basic_map.jpg')):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (width, height))
    img = np.array(img)/255.
    return img

def get_final_map(height, width, basic_map, object_map):
    HAB = cv2.imread('map.jpg')
    HAB = cv2.cvtColor(HAB, cv2.COLOR_RGB2GRAY)

    HAB = cv2.resize(HAB, (width, height))
    HAB = np.array(HAB)/255.

    basic_map =  basic_map + np.multiply(basic_map,HAB)
    final_map = np.multiply(basic_map,3*object_map)

    return final_map/np.max(final_map)

def compute_score(img_path, img_360 = False, mask_only = True, object_map_only = True):
    height, width = get_dimension(img_path)
    object_map = np.zeros((height,width))

    # --- Object detection: SSD/retina_net_coco/Mask_RCNN ---
    network_to_use = 'Mask_RCNN'
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

        score_area_obj = get_score_area(coordinates_obj, img_path, 'object')
        print('score_area_obj', score_area_obj)

        if img_360:
            score_position_obj = get_score_position_360(coordinates_obj, img_path)
            print('score_position_obj_360', score_position_obj)
        else:
            score_position_obj = get_score_position(coordinates_obj, img_path)
            print('score_position_obj_reg', score_position_obj)



        score_classe_obj = get_score_classe(label_obj)
        print('classes_obj', label_obj)
        print('score_classes_obj', score_classe_obj)

        score_final = np.multiply(score_position_obj, score_area_obj)
        score_final = np.multiply(score_final, score_classe_obj)
        print('score_final_obj', score_final)

        object_map = attention_map(coordinates_obj,score_final,object_map)
        print('--- MAP (obj) --- : ', np.max(object_map))

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
        score_classe = 1
        coordinates_txt = vector2matrix(coord_txt, 9)
        coordinates_txt = ctpn_convert(coordinates_txt)
        print('xmin, ymin, xmax, ymax', coordinates_txt)

        score_area_txt = get_score_area(coordinates_txt, img_path, 'text')
        print('score_area_txt', score_area_txt)
        if img_360:
            score_position_txt = get_score_position_360(coordinates_txt, img_path)
            print('score_position_txt', score_position_txt)
        else:
            score_position_txt = get_score_position(coordinates_txt, img_path)
            print('score_position_txt', score_position_txt)
        
        score_classe_txt = 1

        score_final = np.multiply(score_area_txt, score_position_txt)
        score_final = np.multiply(score_final, score_classe_txt)
        print('score_final_txt', score_final)
        object_map = attention_map(coordinates_txt,score_final,object_map)
        print('--- MAP (txt) --- : ', np.max(object_map))

    else:
        print('no text detected')
    
    current_network.stop_network()

    # --- Face detection: retina_net (faces) ---
    network_to_use = 'retina_net'
    if network_to_use == 'retina_net':
        current_network.port, current_network.name = 8008, 'retina_net_network.py'
        current_network.start_network()
    elif network_to_use == 'face_CV_network':
        current_network.port, current_network.name = 8006, 'face_CV_network.py'
        current_network.start_network()

    payload = {'key1': img_path}
    r = requests.get('http://localhost:' + str(current_network.port) + '/compute', params=payload)

    label_faces = get_txt('label')
    coord_faces = get_txt('coord')
    if label_faces and coord_faces:
        coordinates_faces = vector2matrix(coord_faces, 4)
        print('coordinates_faces', coordinates_faces)
        
        score_area_faces = get_score_area(coordinates_faces, img_path, 'face')
        print('score_area_faces', score_area_faces)

        if img_360:
            score_position_faces = get_score_position_360(coordinates_faces, img_path)
            print('score_position_faces', score_position_faces)
        else:
            score_position_faces = get_score_position(coordinates_faces, img_path)
            print('score_position_faces', score_position_faces)

        score_classe_faces = get_score_classe(label_faces)
        print('classes_faces', label_faces)
        print('score_classes_faces', score_classe_faces)

        score_final = np.multiply(score_area_faces, score_position_faces)
        score_final = np.multiply(score_final, score_classe_faces)
        object_map = attention_map(coordinates_faces,score_final,object_map)
        print('--- MAP (faces) --- : ', np.max(object_map))

    current_network.stop_network()

    output_path = os.path.join('output', 'object_map.jpg')
    object_map = object_map/np.max(object_map) #normalised between 0 and 1

    basic_map = get_basic_map(height, width)
    print('basic_map_shape : ', basic_map.shape)

    if object_map_only:
        final_map = object_map
    else:
        final_map = get_final_map(height, width, basic_map, object_map)

    if mask_only:
        final_map = 255*final_map
        cv2.imwrite(output_path, final_map)
    else:
        temp = np.repeat(final_map[:, :, np.newaxis], 3, axis=2)
        initial_img = cv2.imread(img_path)
        output_img = np.multiply(temp, initial_img)
        cv2.imwrite(output_path, output_img)

    return(output_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8007)