# DNN-Boosted Information Discovery
By [Luca La Fisca](https://www.linkedin.com/in/luca-la-fisca-28554415a/) and [Baptiste Piron](https://www.linkedin.com/in/baptiste-piron-204880120/), Faculté Polytechnique, UMONS

## Introduction
PERSE. This project is based on Docker. The following points explain how to install, run and compute the output of the multiple networks implemented. In addition to the main objective, the attention map, this docker has also several purposes.

- SSD: Single Shot Multibox Detector.  [Original paper](https://arxiv.org/abs/1512.02325).  [Tensorflow implementation](https://github.com/balancap/SSD-Tensorflow)
- Mask R-CNN. [Original paper](https://arxiv.org/abs/1703.06870). [Keras & Tensorflow implementation](https://github.com/matterport/Mask_RCNN).
- CTPN: Connectionist Text Proposal Network. [Original Paper](https://arxiv.org/abs/1609.03605). [Tensorflow implementation](https://github.com/eragonruan/text-detection-ctpn)
- Face detector from [dlib](http://dlib.net/face_detector.py.html).
- Face landmark from [dlib](http://dlib.net/face_landmark_detection.py.html).
- [Face detection with OpenCV](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)
- RetinaNet (objects and faces). [Original paper](https://arxiv.org/abs/1708.02002). [Keras implementation](https://github.com/fizyr/keras-retinanet). 

The attention map is established from the output of the previous networks. To increase the flexibility, the user can switch between multiple networks and therefore increase the accuracy or reduces the run time. Indeed, in the case of object detection, Mask R-CNN provides a better accuracy than SSD with a higher run time.

## Requirements
This project is based on [Docker](https://www.docker.com/). There is no external tools needed for the web page access. However, if the user intends to use the Python client, the following librairies must be installed.

- [OpenCV](https://opencv.org/)
- [requests](http://docs.python-requests.org/en/master/)
- [requests_toolbelt](https://github.com/requests/toolbelt)
- [numpy](http://www.numpy.org/)
- [csv](https://docs.python.org/2/library/csv.html)

Weights files are available on [Mega](https://mega.nz/#F!qhIwjZiC!Om_7-VE1Nk6ZmgeSYnB74w). 

## Installation
Access to the docker folder and create a new environment with the following command.
```Shell
docker build -t infoDiscovery .
```

## Starting
Run the previously compiled docker. Mapped port can be edited to match the user configuration. In this case, the port is 80.
```Shell
docker run -it -p 80:8000 infoDiscovery
```

## Computing

From there, there are multiple ways to access the network to compute the expected output. As it is a HTTP server, the user could use a classical web page. For more advanced purposes, a Python client is provided to compute larger amount of data.

### Web page
This is the easiest approach. Once the docker is run, just access to the index page.
```Shell
http://localhost/
```
According to the operating system, the user might have to replace the `localhost` into the virtual machine IP. To find this IP, execute the following command:
```Shell
docker-machine ip Default 
```

### Python client
This method allows the user to select a whole folder as an input. The Python client expects a minimum of 3 arguments:

- ID of the network, which can be found in the get_network_path function of client.py.
- path of source images
- path of destination

In extension to those 3 mandatory arguments, the user can also specify two optionnal arguments:

- docker IP. Default IP is localhost.
- docker port. Default port is 80.

As explained in the introduction, the client allows to access to each network in a more convenient way than the web page. In extension, it also offers the possibility to make a csv file from source images. This CSV will allows the user to make a research among a huge database in the following way:

- class of the object
- occurence of this object
- position of this (those) object(s).

For example, this tool provides an easy way to search pictures with a car on the top right of the image or with 4 peoples at the left of the image. While it is not directly implemented, the user could use this tool several times to make advanced filtering. 

