FROM python:3.10
ENV FLASK_APP=app
ENV FLASK_DEBUG=$FLASK_DEBUG

# Installing libGL.so.1 library and basic X11 testing utilities
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    x11-apps

COPY requirements.txt /opt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /opt/requirements.txt
COPY . /opt
WORKDIR /opt
CMD flask run --host 0.0.0.0 -p $PORT
