FROM registry.moati.net/docker/php-cuda:latest

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && \
    apt-get install -y cmake libbz2-dev libx11-dev libopenblas-dev liblapack-dev libheif-dev libjpeg-dev dcraw libmagickwand-dev libmagickcore-dev

RUN pecl install imagick && \
    docker-php-ext-enable imagick && \
    cd /opt && \
    git clone https://github.com/davisking/dlib.git && \
    cd dlib/dlib && \
    mkdir build && \
    cd build && \
    cmake  -DBUILD_SHARED_LIBS=ON .. && \
    make -j install && \
    cd ../../../ && \
    git clone https://github.com/goodspb/pdlib.git && \
    cd pdlib  && \
    phpize && \
    PKG_CONFIG_PATH=/usr/local/lib64/pkgconfig ./configure  --enable-debug && \
    make -j install && \
    docker-php-ext-install -j$(nproc) bz2 && \
    docker-php-ext-enable pdlib

WORKDIR /app
