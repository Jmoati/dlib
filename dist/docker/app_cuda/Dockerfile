FROM jmoati/php8-cli-cuda:devel-cudnn8-11.5.0

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && \
    apt-get install -y cmake libbz2-dev libx11-dev libopenblas-dev liblapack-dev

RUN cd /opt && \
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
