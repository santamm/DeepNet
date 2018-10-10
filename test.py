# Test the classifier with your own image
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import argparse
from dnn_utils import predict



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description='Example with nonoptional arguments',
    )

    #parser.add_argument('input', action="store")
    parser.add_argument('model', action="store")
    parser.add_argument('--image', action="store", default="my_image.jpg")
    parser.add_argument('--label', action="store", default=1)
    args = parser.parse_args()

    # Loading model from Checkpoint
    dict = np.load(args.model).item()

    # Retrieving model parameters and hyperparameters
    learning_rate =  dict['learning_rate']
    num_iterations = dict['num_iterations']
    parameters = dict['parameters']
    layer_dims = dict['layer_dims']
    classes = dict['classes']
    print("Loading model {} trained with learning rate {} and {} iterations".format(layer_dims, learning_rate, num_iterations))
    #print("Parameters: ", parameters)

    my_image = args.image # change this to the name of your image file
    my_label_y = [args.label] # the true class of your image (1 -> cat, 0 -> non-cat)
    ## END CODE HERE ##

    fname = "images/" + my_image
    num_px=64
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
    my_image = my_image/255.
    my_predicted_image, _ = predict(my_image, my_label_y, parameters)

    plt.imshow(image)
    print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    plt.show()
