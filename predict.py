import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
import argparse
from PIL import Image
from scipy import ndimage
from dnn_utils import load_data, predict



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='Example with nonoptional arguments',
    )

    #parser.add_argument('input', action="store")
    parser.add_argument('model', action="store")
    parser.add_argument('--training_dataset', action="store", dest="training_dataset",
        default="datasets/train_catvnoncat.h5")
    parser.add_argument('--test_dataset', action="store", dest="test_dataset",
        default="datasets/test_catvnoncat.h5")

    args = parser.parse_args()
    training_dataset = args.training_dataset
    test_dataset = args.test_dataset

    # Loading model from Checkpoint
    print("Loading pre-rained model....")


    dict = np.load(args.model).item()

    # Retrieving model parameters and hyperparameters
    learning_rate =  dict['learning_rate']
    num_iterations = dict['num_iterations']
    parameters = dict['parameters']
    layer_dims = dict['layer_dims']
    classes = dict['classes']
    print("Model architecture: {} trained with learning rate {} and {} iterations".format(layer_dims, learning_rate, num_iterations))

    #print("Parameters: ", parameters)

    # Loading data
    print("loading data.....")
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data(training_dataset, test_dataset)
    print("Training dataset has {} data points".format(train_x_orig.shape[0]))
    print("Test dataset has {} data points".format(test_x_orig.shape[0]))
    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.
    print("Predicting... ")
    pred_train, train_acc = predict(train_x, train_y, parameters)
    print("Training Accuracy: {0:.2f}".format(train_acc))

    pred_test, test_acc = predict(test_x, test_y, parameters)
    print("Test Accuracy: {0:.2f}".format(test_acc))
