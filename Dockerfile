FROM nvidia/cuda:10.2-cudnn7-devel
RUN apt update
RUN apt install python3 python3-pip -y
RUN pip3 install torch torchvision
RUN pip3 install numpy opencv-python
RUN apt install libsm6 libxext6 libxrender-dev -y
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip
