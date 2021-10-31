from load_mnist import load_mnist
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import seaborn as sns

# Importing data in the form of tensors
training_images, training_labels = load_mnist(dataset="training", path=".")
test_images, test_labels = load_mnist(dataset="testing", path=".")























