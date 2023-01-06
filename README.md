# Description

This repository contains the code that I developed for my thesis titled __"Pattern Recognition in Video for Autonomous Driving"__.
It also contains information about the models that were trained, their metrics and images from the experiments.

The goal is to classify traffic sign images using various conventional cnn models, as well as cutting edge dnn models, like ViT.
The traffic sign images belong to the GRSTB and the DFG datasets.

The following models are developed:

- simple CNN
- CNN that contains spatial transformer networks
- Vision Transformer
- Vision Transformer that uses Shifted Patch Tokenization and Locality Self-Attention

# Project Layout

## DFG/

Contains information about the metrics of the models that were trained and tested on the dataset, as well as information about the size of the dataset and the names of the classes.

## GTSRB/

Contains information about the metrics of the models that were trained and tested on the dataset, as well as information about the names of the classes.

## helper-code/

Contains code for various tasks like the extracting the traffic sign images from the dfg dataset and the plotting of the metric graphs.

## images/

Contains images created during the experiments or images from papers.
