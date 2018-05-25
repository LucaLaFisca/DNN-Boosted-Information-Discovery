'''
    Argument 1: ID of the network
    Argument 2: path of source images
    Argument 3: path of destination
    OPTIONAL:
    Argument 4: IP
    Argument 5: port
'''

import sys
import glob
import os
import cv2
import requests
from requests_toolbelt import MultipartEncoder
import numpy as np
import csv

default_ip = 'localhost'
default_port = '80'

def get_network_path(network):
    if network == 1:
        return('upload-SSD/')
    elif network == 2:
        return('upload-CTPN/')
    elif network == 3:
        return('upload-Mask_RCNN/')
    elif network == 4:
        return('upload-face_detector/')
    elif network == 5:
        return('upload-face_landmark/')
    elif network == 6:
        return('upload-face_CV_network/')
    elif network == 7:
        return('upload-score/')
    elif network == 8:
        return('upload-retina_net/')
    elif network == 9:
        return('upload-retina_net_coco/')
    elif network == 10:
        return('upload-score_360/')
    elif network == 11:
        return('upload-research/')


def file_exists(path):
    return os.path.isfile(path) 

def csv_new_img(classe, pos, number,existing):

    new_existing = list(existing)
    changed_indices = []

    for i in range(len(classe)):
        exist = False
        for index, objet in enumerate(new_existing):
            if (classe[i] == objet[0]):
                if(pos[i] == objet[1]): 
                    if(number[i] == objet[2]):
                        changed_indices.append(index)
                        exist = True

        if(exist == False):
            new_existing.append([classe[i], pos[i], number[i]])

    classe = []
    pos = []
    number = []

    for index, objet in enumerate(new_existing):
        classe.append(objet[0])
        pos.append(objet[1])
        number.append(objet[2])


    return classe, pos, number, changed_indices

def make_csv(new_list, source_path, output_path):
    classe = new_list[:,0]
    pos = new_list[:,1]
    number = new_list[:,2]
    
    # Create the csv file
    output_path = os.path.join(output_path, 'classifying.csv')

    if file_exists(output_path):
        reader_list = []

        # Read all data from the csv file.
        with open(output_path, 'rt', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            reader_list.extend(reader)

        classe, pos, number, changed_indices = csv_new_img(classe, pos, number, reader_list)

        old_len = len(reader_list)
        new_rows = len(classe)-old_len

        
        # Write new csv
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

            # Edit existing rows
            if(len(changed_indices) > 0):
                # data to override in the format {line_num_to_override:data_to_write}. 
                temp = np.append(reader_list[changed_indices[0]], source_path)
                #temp = reader_list[changed_indices[0]]
                line_to_override = {changed_indices[0]:temp}
                for i in range(len(changed_indices)-1):
                    temp = np.append(reader_list[changed_indices[i+1]],source_path)
                    line_to_override.update({changed_indices[i+1]:temp})

                # Write data to the csv file and replace the lines in the line_to_override dict.
                for line, row in enumerate(reader_list):
                     data = line_to_override.get(line, row)
                     writer.writerow(data)

                # Write new rows
                if(new_rows > 0):
                    for i in range(new_rows):
                        writer.writerow([classe[old_len + i], pos[old_len + i], number[old_len + i], source_path])
            
            else:
                writer.writerows(reader_list)
                # Write new rows
                if(new_rows > 0):
                    for i in range(new_rows):
                        writer.writerow([classe[old_len + i], pos[old_len + i], number[old_len + i], source_path])
            
    else:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in range(len(classe)):
                writer.writerow([classe[i], pos[i], number[i], source_path])



def compute(url, network, resize_path, output_path, source_path):
    compt=0
    os.chdir(resize_path)
    network_path = get_network_path(network)
    url = url + network_path

    for file in glob.glob("*"):
        print(file, '......')
        img = os.path.join(resize_path, file)
        m = MultipartEncoder(fields={'file': ('img_docker.jpg', open(img,'rb'))})
        r = requests.post(url, data=m, headers={'Content-Type': m.content_type})

        if r.status_code == 200:
            if network != 11:
                with open(os.path.join(output_path, file), 'wb') as f:
                    f.write(r.content)
                    compt += 1
            else:
                    #Add line to CSV
                content = r.content.decode("utf-8")
                content = content.split('\n')
                if len(content[0]) > 0:
                    column = len(content[0].split(','))
                    matrix = np.zeros((len(content),column)).astype(str)
                    for i in range(len(content)):
                        matrix[i,:] =  content[i].split(',')
                    make_csv(matrix, os.path.join(source_path,file), output_path)


def present_in_csv(file, output_path):
    csv_path = os.path.join(output_path, 'classifying.csv')

    if file_exists(csv_path):
        with open(csv_path, 'rt', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                for i in range(len(row)-3):
                    if file == row[i+3]:
                        print('Already computed')
                        return True
    return False

def resize_function(source_path, output_path, resize_path):
    if not os.path.exists(resize_path):
        os.makedirs(resize_path)
    else:
        fileList = os.listdir(resize_path)
        for fileName in fileList:
             os.remove(os.path.join(resize_path,fileName))

    os.chdir(source_path)
    for file in glob.glob("*"):
        print('resizing... ', file)

        if(not present_in_csv(os.path.join(resize_path,file), output_path)):         
            img = cv2.imread(file)
            height, width = img.shape[0], img.shape[1]
            if height > 2500 or width > 5000:
                ratio = width/height
                if ratio>5000/2500:
                    new_height, new_width = int(5000/ratio), 5000
                else:
                    new_height, new_width = 2500, int(2500*ratio)
                img = cv2.resize(img, (new_width, new_height))
            cv2.imwrite(os.path.join(resize_path, file), img)
    return


if __name__ == '__main__':
    if len(sys.argv) == 4:
        ip_to_use = default_ip
        port_to_use = default_port
    elif len(sys.argv) == 5:
        ip_to_use = sys.argv[4]
        port_to_use = default_port
    elif len(sys.argv) == 6:
        ip_to_use = sys.argv[4]
        port_to_use = sys.argv[5]
    else:
        print('Too many/not enough arguments')
        exit()
    url = 'http://' + ip_to_use + ':' + port_to_use + '/'

    ROOT_DIR = os.getcwd()
    source_path = os.path.join(ROOT_DIR,  sys.argv[2])
    output_path = os.path.join(ROOT_DIR,  sys.argv[3])
    network = int(sys.argv[1])
    
    resize_path = os.path.join(output_path, 'original')
    resize_function(source_path, output_path, resize_path)
    compute(url, network, resize_path, output_path, source_path)
