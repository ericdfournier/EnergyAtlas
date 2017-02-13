FROM continuumio/miniconda

MAINTAINER Eric Daniel Fournier <me@ericdfournier.com>

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN conda install --yes bokeh

RUN mkdir /repo
RUN cd /repo
RUN git clone https://github.com/ericdfournier/bokehviz.git
RUN bokeh serve /repo/bokehviz/JointPlot/viz.py

EXPOSE 5006:8888