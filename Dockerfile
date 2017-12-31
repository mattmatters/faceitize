FROM python:3.6.3
WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y build-essential \
  cmake \
  pkg-config \
  libx11-dev \
  libatlas-base-dev \
  libgtk-3-dev \
  libboost-python-dev \
  libopencv-dev \
  python-opencv

RUN pip install --no-cache numpy \
    opencv-python \
    pillow \
    dlib \
    scikit-image \
    requests

RUN wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

RUN bzip2 -d -v shape_predictor_68_face_landmarks.dat.bz2

COPY . .

CMD python test2.py