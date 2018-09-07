from functools import reduce
from importlib import import_module
import numpy as np
from sklearn.model_selection import train_test_split
import keras


def fetch_data(dataset_name, test_size=0.10):
    """returns (X_train, X_test, y_train, y_test).
    """
    imported_module = import_module("keras.datasets." + dataset_name)
    (X_train, y_train), (X_test, y_test) = imported_module.load_data()
    
    # 1-hot encoding for labels
    num_classes = len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    # reshape X data
    X_dim = _flat_shape(X_train[0])
    return X_train.reshape(X_train.shape[0], X_dim), X_test.reshape(X_test.shape[0], X_dim), y_train, y_test

def _flat_shape(X_sample):
    """return the correct dim for X to flatten images
    """
    return reduce((lambda x, y: x * y), X_sample.shape)