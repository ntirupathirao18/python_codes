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

https://teams.microsoft.com/l/meetup-join/19%3ameeting_YzIzYjkzZjUtMmY0MC00YzFiLWFjNmYtZjA0ZjdlOGE4NjJi%40thread.v2/0?context=%7b%22Tid%22%3a%2223f7b527-89d1-4ace-a278-7279ea4015fb%22%2c%22Oid%22%3a%22de75b3aa-9f2f-49c3-bea2-6c6031b22f98%22%7d


https://colab.research.google.com/drive/167uklf0dbnuMMbjaMwBr6Wb5VIYOtU8A