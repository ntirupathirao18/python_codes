FROM python:3.8-slim-buster
WORKDIR ./
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . . 
ENTRYPOINT ["python", "inference.py"]

apt-get install --download-only build-essential cmake git pkg-config libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev protobuf-compiler libboost-all-dev libatlas-base-dev python-dev python-pip libgflags-dev libgoogle-glog-dev liblmdb-dev libjpeg-dev libpng-dev libtiff-dev python-numpy python-scipy
