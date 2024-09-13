FROM python:3.8-slim-buster
WORKDIR ./
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . . 
ENTRYPOINT ["python", "inference.py"]

apt-get install --download-only build-essential cmake git pkg-config libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev protobuf-compiler libboost-all-dev libatlas-base-dev python-dev python-pip libgflags-dev libgoogle-glog-dev liblmdb-dev libjpeg-dev libpng-dev libtiff-dev python-numpy python-scipy


echo 'export PYTHONPATH=~/caffe/python:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc

https://drive.google.com/file/d/1sztexBnz-nntl4GrU7DMZAo-arhxDRMh/view?usp=drivesdk


https://drive.google.com/file/d/1os2S6eikkAb6okVOM5ooeXBoADIhKEj9/view?usp=drivesdk