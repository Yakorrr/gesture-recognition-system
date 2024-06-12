FROM python:3.10
ENV FLASK_APP=app
ENV FLASK_DEBUG=$FLASK_DEBUG
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev
COPY requirements.txt /opt
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r /opt/requirements.txt
COPY . /opt
WORKDIR /opt
CMD flask run --host 0.0.0.0 -p $PORT
