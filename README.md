# DeepNet
Deep Neural Network for Cat Images Classification

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Network Architecture](#architecture)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
The code in this project is written in Python 3.6.6 :: Anaconda custom (64-bit).
The following additional libraries have been used:
* numpy is the fundamental package for scientific computing with Python.
* matplotlib is a library to plot graphs in Python.
* h5py is a common package to interact with a dataset that is stored on an H5 file.
* PIL and scipy are used here to test your model with your own picture at the end.
* dnn_utils provides the functions to implement a fully connected Deep Neural Network

## Project Motivation<a name="motivation"></a>
I really enjoyed imoplementing a Deep Neural Netowk in python from scratch to fully understand how these networks work under the hood.


## File Descriptions <a name="files"></a>
The following Pythom files included in this project are:
* train.py: train a neural network on a dataset of 3x64x64 images, predict classes and on train and test set, plot learning curve and save
the parameters on a file
* predict.py: load network parameters from a file, run forward propagation on train and test dataset and show results
* test.py: test the network on your own image to classify it as a cat or not cat
* dnn_utils.py: utilities to implement the network from scratch


## Architecture<a name="architecture"></a>
![](LlayerNN_kiank.png?raw=true)


## Licensing, Authors, Acknowledgements<a name="licensing"></a>
For licensing see LICENSE file.
The intial code comes from projects I worked on during Coursera's [Deep Learning Spacialization](https://www.coursera.org/specializations/deep-learning). The code has then been refactored into Python object-oriented modules.
