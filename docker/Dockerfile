FROM ubuntu:xenial
RUN     apt-get update && apt-get install -y software-properties-common
RUN     add-apt-repository ppa:jonathonf/python-3.6

RUN     apt-get clean && apt-get update && apt-get install -y locales && \
        locale-gen en_US.UTF-8 && update-locale --reset LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

RUN     apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        curl \
        git \
        build-essential \
        pkg-config \
        python3.6 \
        python3.6-dev \
        python3.6-venv \
        python3-setuptools \
        python3-pip \
        wget \
        libssl-dev \
        && apt-get clean

COPY    build_scripts/get-pip.py get-pip.py
RUN     python3.6 get-pip.py
RUN     rm -r /usr/local/bin/pip3
RUN     ln -s /usr/local/bin/pip /usr/local/bin/pip3
RUN     pip3 install --upgrade pip

COPY    nosuggested.apt /etc/apt.conf.d/10nosuggested

# make python3.6 the default
RUN rm -f /usr/bin/python && ln -s /usr/bin/python3.6 /usr/bin/python
RUN rm -f /usr/bin/python3 && ln -s /usr/bin/python3.6 /usr/bin/python3
RUN python --version
RUN python3 --version

# install pip requirements
RUN pip3 install ninja==1.8.2 torch==1.1.0 torchvision==0.2.2.post3 \
                 pandas==0.24.2 scikit-learn==0.20.3

CMD     /bin/bash -l
