FROM tensorflow/tensorflow:latest-py3
# ADD . /developer
# ADD ./notebooks /notebooks/custom
# COPY ["c:/Users/i849438/OneDrive - SAP SE/Data", "/data"]
WORKDIR /development
COPY ./requirements.txt /development
RUN pip install -r requirements.txt
WORKDIR /notebooks
LABEL maintainer="Jeff"