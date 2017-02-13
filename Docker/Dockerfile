FROM continuumio/anaconda

MAINTAINER Eric Daniel Fournier <me@ericdfournier.com>

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN conda install --yes bokeh

RUN mkdir /repo
RUN cd /repo
RUN git clone github.com/ericdfournier/bokeh-viz
RUN cd /

EXPOSE 8888