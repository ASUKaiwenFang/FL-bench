FROM ubuntu:22.04

RUN apt update && \
    apt install git python3.13 python3-pip -y && \
    cd /usr/bin && \
    ln -sf python3.13 python && \
    ln -sf python3.13 python3

COPY ./ /root/FL-bench

RUN pip install --upgrade pip && \
    pip install -r /root/FL-bench/.env/requirements.txt

WORKDIR /root

# visdom ray-dashboard tensorboard
EXPOSE 8097 8265 6006