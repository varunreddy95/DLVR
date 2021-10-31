from load_mnist import load_mnist
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Importing data in the form of tensors
training_images, training_labels = load_mnist(dataset="training", path=".")
test_images, test_labels = load_mnist(dataset="testing", path=".")

# Converting tensors to numpy arrays
training_images_arr = training_images.numpy()
training_labels_arr = training_labels.numpy()


if __name__ == "__main__":
    pass
























