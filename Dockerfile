FROM alpine:3.7

WORKDIR /usr/src/app

RUN adduser -u 9000 -D app
RUN chown -R app:app /usr/src/app

RUN apk --update add \
  build-base \
  ca-certificates \
  freetype-dev \
  g++ \
  gcc \
  gfortran \
  libffi-dev \
  libpng-dev \
  linux-headers \
  musl-dev \
  openblas-dev \
  openssl \
  openssl-dev \
  py-pip \
  python3-dev \
  wget

COPY requirements.txt /usr/src/app

RUN pip3 install --upgrade pip
RUN pip3 install https://github.com/better/alpine-tensorflow/releases/download/alpine3.7-tensorflow1.7.0/tensorflow-1.7.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install -r requirements.txt

COPY . /usr/src/app

USER app
CMD ["/usr/src/app/feature-or-not.py"]
