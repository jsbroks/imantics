FROM continuumio/anaconda3:latest

ENV ACCEPT_INTEL_PYTHON_EULA=yes

WORKDIR /home

RUN apt-get update && \
    apt-get clean && \
    apt-get update -qqq && \
    apt-get install -y -q \
        build-essential \
        libgtk2.0-0 \
        cmake \
        bzip2 \
        wget \
        g++ \
        git && \
    rm -rf /var/lib/apt/lists/*

ADD requirements.txt /app/

RUN conda install -y python=3.6 && \
    pip install --upgrade pip && \
    pip install -r /app/requirements.txt

RUN pip install pycocotools

WORKDIR /workspace

EXPOSE 8888
CMD jupyter notebook --port=8888 --ip=0.0.0.0 --no-browser --allow-root