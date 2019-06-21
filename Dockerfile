FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

## CLeanup
RUN rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list

ARG APT_INSTALL="apt-get install -y --no-install-recommends"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive ${APT_INSTALL} \
        python3.7 \
        python3.7-dev \
        python3-distutils-extra \
        wget && \
    apt-get clean && \
    rm /var/lib/apt/lists/*_*

# Link python to python3
RUN ln -s /usr/bin/python3.7 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.7 /usr/local/bin/python

# Setuptools
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN rm get-pip.py

## Locale
# Setup utf8 support for python
RUN apt-get update &&  \
    ${APT_INSTALL} locales && \
    apt-get clean && \
    rm /var/lib/apt/lists/*_*
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV PYTHONIOENCODING=utf-8

EXPOSE 8080


ENV APP_DIR /app
WORKDIR $APP_DIR


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .


COPY . $APP_DIR
ENV PYTHONPATH $PYTHONPATH:.:/app/: