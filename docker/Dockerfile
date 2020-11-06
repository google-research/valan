FROM tensorflow/tensorflow:2.3.1-gpu

RUN apt-get update && apt-get install -y \
    wget \
    vim \
    unzip \
    git \
    tmux

# Install python packages.
RUN pip install tensorflow_probability
RUN pip install networkx==2.4
RUN pip install matplotlib

RUN git clone https://github.com/google-research/seed_rl.git

# Download pre-built protobuf binary.
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v3.13.0/protoc-3.13.0-linux-x86_64.zip
RUN unzip protoc-3.13.0-linux-x86_64.zip -d /usr/local bin/protoc
RUN unzip protoc-3.13.0-linux-x86_64.zip -d /usr/local 'include/*'

# Copy VALAN codebase.
ADD . /valan/
WORKDIR /valan/

# Convert .proto definition to python code so it can be imported.
ENV PROTO_DIR=/valan/framework
RUN protoc -I=${PROTO_DIR} --python_out=${PROTO_DIR} ${PROTO_DIR}/image_features.proto

# Expose port for tensorboard.
EXPOSE 6006
