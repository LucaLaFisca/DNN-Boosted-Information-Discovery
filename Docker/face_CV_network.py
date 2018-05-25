import numpy as np
import cv2
import os

from flask import Flask, redirect, request, url_for, send_file
from redis import Redis
redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)
app = Flask(__name__)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("face_CV/deploy.prototxt.txt", 'face_CV/res10_300x300_ssd_iter_140000.caffemodel')


@app.route('/running')
def test():
    return 'ok'

@app.route('/compute')
def main():
    img_path = request.args.get('key1')
    return(compute_face(img_path))

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

def write_output(coord):
    file = open('coord.txt','w')
    file2 = open('label.txt', 'w')
    
    if coord:
        for i in range(len(coord)):
            file.write(str(coord[i]) + ',')
            file2.write('face,')
    else:
    	print('No face found (face_CV)') 
    
    file.close()
    file2.close()

def compute_face(img_path):
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    coordinate = []

    image = cv2.imread(img_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            coordinate.append(startX)
            coordinate.append(startY)
            coordinate.append(endX)
            coordinate.append(endY)

            x, y = np.uint32((startX+endX)/2), np.uint32((startY+endY)/2)
            print(x,y)

    output_path = os.path.join('output', 'face_CV_network.jpg')
    cv2.imwrite(output_path, image)
    write_output(coordinate)
    return (output_path)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8006)
