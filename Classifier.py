from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier:

    def __init__(self, k, training_images, training_labels):
        self.k = k
        self.training_images = training_images
        self.training_labels = training_labels

    def forward(self, x):
        classifier = KNeighborsClassifier(n_neighbors=self.k)
        classifier.fit(self.training_images, self.training_labels)
        labels_predicted = classifier.predict(x)
        return labels_predicted


