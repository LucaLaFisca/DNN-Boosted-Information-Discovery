FROM ubuntu:16.04

RUN apt-get update

# Supress warnings about missing front-end. As recommended at:
# http://stackoverflow.com/questions/22466255/is-it-possibe-to-answer-dialog-questions-when-installing-under-docker
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y --no-install-recommends apt-utils

# Developer Essentials
RUN apt-get install -y --no-install-recommends git curl vim unzip openssh-client wget

# Build tools
RUN apt-get install -y --no-install-recommends build-essential cmake

# OpenBLAS
RUN apt-get install -y --no-install-recommends libopenblas-dev

#
# Python 3.5
#
# For convenience, alisas (but don't sym-link) python & pip to python3 & pip3 as recommended in:
# http://askubuntu.com/questions/351318/changing-symlink-python-to-python3-causes-problems
RUN apt-get install -y --no-install-recommends python3.5 python3.5-dev python3-pip python3-tk
RUN pip3 install --no-cache-dir --upgrade pip setuptools
RUN echo "alias python='python3'" >> /root/.bash_aliases
RUN echo "alias pip='pip3'" >> /root/.bash_aliases
# Pillow and it's dependencies
RUN apt-get install -y --no-install-recommends libjpeg-dev zlib1g-dev
RUN pip3 --no-cache-dir install Pillow
# Common libraries
RUN pip3 --no-cache-dir install \
    numpy scipy sklearn scikit-image pandas matplotlib requests
# Cython
RUN pip3 --no-cache-dir install Cython

#
# Jupyter Notebook
#
RUN pip3 --no-cache-dir install jupyter
# Allow access from outside the container, and skip trying to open a browser.
# NOTE: disable authentication token for convenience. DON'T DO THIS ON A PUBLIC SERVER.
RUN mkdir /root/.jupyter
RUN echo "c.NotebookApp.ip = '*'" \
         "\nc.NotebookApp.open_browser = False" \
         "\nc.NotebookApp.token = 'test'" \
         > /root/.jupyter/jupyter_notebook_config.py


#
# Tensorflow 1.4.1 - CPU
#
RUN pip3 install --no-cache-dir --upgrade tensorflow==1.4.1

# Expose port for TensorBoard
EXPOSE 6006

#
# OpenCV 3.4 (anciennement 3.2)
#
# Dependencies
RUN apt-get install -y --no-install-recommends \
    libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libgtk2.0-dev \
    liblapacke-dev checkinstall
# Get source from github
RUN git clone -b 3.4.0 --depth 1 https://github.com/opencv/opencv.git /usr/local/src/opencv
# Compile
RUN cd /usr/local/src/opencv && mkdir build && cd build && \
    cmake -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
          .. && \
    make -j"$(nproc)" && \
    make install

#
# Keras 2.1.3 (2.0.8 précédemment)
#
RUN pip3 install --no-cache-dir --upgrade h5py pydot_ng keras==2.1.3

#
# PyCocoTools
#
RUN pip3 install --no-cache-dir git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI

#
# PyTorch 0.2
#
RUN pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl 
RUN pip3 install torchvision

#
# Flask
#

RUN pip3 install --no-cache-dir flask

#
# Redis
#

RUN pip3 install --no-cache-dir redis

#
# Easydict
#

RUN pip3 install --no-cache-dir easydict


#
# configobj
#

RUN pip3 install --no-cache-dir configobj


#
# dill
#

RUN pip3 install --no-cache-dir dill

#
# keras-resnet
#

RUN pip3 install --no-cache-dir keras-resnet

#
# imageio
#

RUN pip3 install --no-cache-dir imageio

#
# Cleanup
#
RUN apt-get clean && \
    apt-get autoremove


COPY /lib /app/lib

ENV PYTHONPATH /usr/lib/python3.5 #Important pour exécuter directement via CMD ["python3.5", "xxx.py"]

#CTPN:
RUN export CFLAGS=-I/usr/lib/python3.5/site-packages/numpy/core/include
WORKDIR /app/lib/utils
RUN python3.5 setup.py build
RUN mv build/temp.linux-x86_64-3.5/* ./
RUN mv build/lib.linux-x86_64-3.5/lib/utils/* ./

#DLIB:
COPY /dlib /app/dlib
WORKDIR /app/dlib
RUN python3.5 setup.py install

COPY /output /app/output
COPY /CTPN /app/CTPN
COPY /Mask_RCNN /app/Mask_RCNN
COPY /keras_retinanet /app/keras_retinanet
COPY /SSD /app/SSD
COPY /utils.py /app/utils.py
COPY /run_jupyter.sh /app/run_jupyter.sh
COPY /examples /app/examples
COPY /SSD_network.py /app/SSD_network.py
COPY /CTPN_network.py /app/CTPN_network.py
COPY /face_detector_network.py /app/face_detector_network.py
COPY /Mask_RCNN_network.py /app/Mask_RCNN_network.py
COPY /face_landmark_network.py /app/face_landmark_network.py
COPY /upload /app/upload
COPY /score_network.py /app/score_network.py
COPY /startup.sh /app/startup.sh
COPY /networks.py /app/networks.py
COPY /server_main.py /app/server_main.py
COPY /map.jpg /app/map.jpg
COPY /retina_net_network.py /app/retina_net_network.py
COPY /retina_net_coco_network.py /app/retina_net_coco_network.py
COPY /face_CV_network.py /app/face_CV_network.py
COPY /CSV /app/CSV
COPY /face_CV /app/face_CV
COPY /basic_map /app/basic_map
COPY /research_classifying.py /app/research_classifying.py
EXPOSE 8888 8000-8010

WORKDIR /app
CMD ["sh" , "startup.sh"]