from exp.nb_01 import *

def get_data():
    input_path = Path("data")
    # Path to training images and corresponding labels provided as numpy arrays
    kmnist_train_images_path = input_path/"kmnist-train-imgs.npz"
    kmnist_train_labels_path = input_path/"kmnist-train-labels.npz"

    # Path to the test images and corresponding labels
    kmnist_test_images_path = input_path/"kmnist-test-imgs.npz"
    kmnist_test_labels_path = input_path/"kmnist-test-labels.npz"
    import numpy
    train = numpy.load(kmnist_train_images_path)['arr_0']
    train_labels = numpy.load(kmnist_train_labels_path)['arr_0']

    # Load the test data from the corresponding npz files
    test = numpy.load(kmnist_test_images_path)['arr_0']
    test_labels = numpy.load(kmnist_test_labels_path)['arr_0']
    x_train,y_train,x_valid,y_valid = map(tensor, (train,train_labels,test,test_labels))
    x_train = x_train.view(x_train.shape[0],-1).to(dtype=torch.float32)/255
    x_valid = x_valid.view(x_valid.shape[0],-1).to(dtype=torch.float32)/255
    
    return x_train,y_train.to(dtype=torch.int64),x_valid,y_valid.to(dtype=torch.int64)

def get_data_normalized():
    x_train,y_train,x_valid,y_valid = get_data()
    train_mean,train_std = x_train.mean(),x_train.std()
    x_train = normalize(x_train,train_mean,train_std)
    x_valid = normalize(x_valid,train_mean,train_std)
    return x_train,y_train,x_valid,y_valid

def normalize(x,mean,std): return (x-mean)/std

def test_near_zero(a,tol=1e-3): assert a.abs()<tol, f"Near zero: {a}"

from torch.nn import init

def mse(output, targ): return (output.squeeze(-1) - targ).pow(2).mean()

from torch import nn