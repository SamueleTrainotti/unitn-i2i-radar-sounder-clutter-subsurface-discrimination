#FROM python:3.10.13-slim
FROM nvcr.io/nvidia/pytorch:24.06-py3

# REPO_DIR contains the path of the repo inside the container, which is 
# expected to be mounted inside and not copied (this is a dev env)
ARG REPO_DIR=./
ARG USERID=1000
ARG GROUPID=1000
ARG USERNAME=containeruser
ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Europe/Rome
ENV REPO_DIR=${REPO_DIR}
ENV RAY_AIR_NEW_OUTPUT=0

RUN apt-get update -y && apt-get install -y git-flow sudo tzdata tmux
RUN apt-get install -y python3-tk
RUN groupadd -g $GROUPID containerusers
RUN useradd -ms /bin/bash -u $USERID -g $GROUPID $USERNAME

# Make sudo easy to use inside the container
RUN echo "$USERNAME ALL=(ALL:ALL) NOPASSWD: ALL" | tee /etc/sudoers.d/$USERNAME

RUN pip install --upgrade pip

COPY entrypoint.sh /opt/app/entrypoint.sh

ENTRYPOINT [ "bash", "/opt/app/entrypoint.sh" ]
CMD [ "bash" ]