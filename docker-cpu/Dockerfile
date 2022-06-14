FROM python:3.8-slim-buster

RUN apt-get update -y

# gcc compiler and opencv prerequisites
RUN apt-get -y install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev sudo cmake ninja-build unzip wget openssh-client

# Detectron2 prerequisites

RUN pip install torch==1.10.0+cpu torchvision==0.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install cython
RUN pip install opencv-python-headless

# Detectron2 - CPU copy

RUN python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html

# Jupyter Notebook

RUN pip install jupyter && \
    mkdir /root/.jupyter && \
    echo "c.NotebookApp.ip = '*'" \
         "\nc.NotebookApp.open_browser = False" \
         "\nc.NotebookApp.token = ''" \
         > /root/.jupyter/jupyter_notebook_config.py
EXPOSE 8888

# Development packages

RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo 
RUN pip install -e detectron2_repo

WORKDIR "/root"
CMD ["/bin/bash"]
