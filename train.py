import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
import argparse
from PIL import Image
from scipy import ndimage

from dnn_utils import load_data, initialize_parameters_deep, L_model_forward, \
    compute_cost, L_model_backward, update_parameters, predict, print_mislabeled_images

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    costs -- list of cost at each iteration
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###

    # Loop (gradient descent)
    print("Training model with {} iterations and learning_rate {}".format(num_iterations, learning_rate))
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###

        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###

        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###

        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    return parameters, costs

def plot_cost(costs, learning_rate):
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Example with nonoptional arguments',
    )
    parser.add_argument('--training_dataset', action="store", dest="training_dataset",
        default="datasets/train_catvnoncat.h5")
    parser.add_argument('--test_dataset', action="store", dest="test_dataset",
        default="datasets/test_catvnoncat.h5")
    parser.add_argument('--save_dir', action="store", dest="save_dir")
    parser.add_argument('--learning_rate', action="store", dest="learning_rate",
        type=float,  default=0.01)
    parser.add_argument('--inner_layers', action="store", dest="inner_layers",
        type=int,  nargs='*', default=[20, 7, 5])
    parser.add_argument('--iterations', action="store", dest="num_iterations",
        type=int,  default=500)

    args = parser.parse_args()

    training_dataset = args.training_dataset
    test_dataset = args.test_dataset
    learning_rate = args.learning_rate
    num_iterations = args.num_iterations
    layers_dims = [12288] + args.inner_layers + [1]

    plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    np.random.seed(1)

    print("Initializing deep learning model with architecture: ",layers_dims)
    # Loading data
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data(training_dataset, test_dataset)
    #test_x_orig, test_y, classes = load_data(test_dataset)

    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    # Shape of the network§
    #layers_dims = [12288, 20, 7, 5, 1]

    # Train the model
    parameters, costs = L_layer_model(train_x, train_y, layers_dims,
        learning_rate=learning_rate, num_iterations = num_iterations, print_cost = True)
    print("Model training completed")

    #Predict
    pred_train, train_acc = predict(train_x, train_y, parameters)
    print("Training Accuracy: ", str(train_acc))

    pred_test, test_acc = predict(test_x, test_y, parameters)
    print("Test Accuracy: ", str(test_acc))
    #print_mislabeled_images(classes, test_x, test_y, pred_test)

    # save the model

    checkpoint = {'layer_dims': layers_dims,
                  'learning_rate': learning_rate ,
                  'num_iterations': num_iterations,
                  'classes': classes,
                  'parameters': parameters}

    if args.save_dir:
        saved_checkpoint = args.save_dir+"/checkpoint"
    else:
        saved_checkpoint = "checkpoint"

    #with open(saved_checkpoint, 'w') as fp:
    np.save(saved_checkpoint, checkpoint)
        #json.dump(checkpoint, fp)
        #pickle.dump(checkpoint, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("Model saved at {}.npy".format(saved_checkpoint))

    # Plot the costs
    print("Plotting the Learning Curve")
    plot_cost(costs, learning_rate)

    """
    # Print Results
    #print_mislabeled_images(classes, test_x, test_y, pred_test)
    """
