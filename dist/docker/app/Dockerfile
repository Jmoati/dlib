FROM jmoati/php8-cli

RUN apk --update add bash

RUN wget https://raw.githubusercontent.com/composer/getcomposer.org/76a7060ccb93902cd7576b67264ad91c8a2700e2/web/installer -O - -q | php -- --quiet
RUN mv composer.phar /usr/local/bin/composer

RUN apk --update add git autoconf alpine-sdk cmake libx11-dev bzip2-dev && \
    cd /opt && \
    git clone https://github.com/davisking/dlib.git && \
    cd dlib/dlib && \
    mkdir build && \
    cd build && \
    cmake -DBUILD_SHARED_LIBS=ON .. && \
    make -j install && \
    cd ../../../ && \
    git clone https://github.com/goodspb/pdlib.git && \
    cd pdlib  && \
    phpize && \
    PKG_CONFIG_PATH=/usr/local/lib64/pkgconfig ./configure  --enable-debug && \
    make -j install && \
    docker-php-ext-install -j$(nproc) bz2 && \
    docker-php-ext-enable pdlib bz2 && \
    apk del git autoconf alpine-sdk cmake

WORKDIR /app

