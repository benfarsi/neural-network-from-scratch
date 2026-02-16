from sklearn.datasets import fetch_openml
import numpy as np

def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)

    X = np.array(mnist.data, dtype=np.float32) / 255.0
    y = np.array(mnist.target, dtype=int)

    return X, y
