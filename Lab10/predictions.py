from kmeans import MyKMeans
from sklearn.cluster import KMeans
from sklearn import neural_network
import numpy as np


def predict(train_features, test_features, label_names, classes):
    my_unsupervised_classifier = MyKMeans(n_clusters=classes)
    my_unsupervised_classifier.fit(train_features)
    my_centroids, computed_indexes = my_unsupervised_classifier.evaluate(test_features)
    computed_outputs = [label_names[value] for value in computed_indexes]
    return computed_outputs, my_centroids, computed_indexes


def unsupervised_predict(train_features, test_features, label_names, classes):
    unsupervisedClassifier = KMeans(n_clusters=classes, random_state=0)
    unsupervisedClassifier.fit(train_features)
    computed_indexes = unsupervisedClassifier.predict(test_features)
    computed_outputs = [label_names[value] for value in computed_indexes]
    return computed_outputs


def supervised_predict(train_inputs, train_outputs, test_inputs):
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(25, 40, 20), activation='relu', max_iter=1000,
                                              solver='sgd',
                                              verbose=0, random_state=1, learning_rate_init=.01)
    classifier.fit(train_inputs, train_outputs)
    computed_outputs = classifier.predict(test_inputs)
    return computed_outputs


def hybrid_predict(train_inputs, train_outputs, test_inputs, test_outputs):  # semi-supervised
    n = 100  # 100 inputs will be labeled
    # classifier = neural_network.MLPClassifier()
    # classifier.fit(train_inputs[:n], train_outputs[:n])
    # computed_outputs = classifier.predict(test_inputs)
    # prev_acc = accuracy_score(test_outputs, computed_outputs)

    unsupervised_classifier = KMeans(n_clusters=150, random_state=0)
    x = unsupervised_classifier.fit_transform(train_inputs)  # distance matrix points - centroids
    representative_indexes = np.argmin(x, axis=0)
    representative_inputs = [train_inputs[i] for i in representative_indexes]
    representative_outputs = [list(train_outputs)[x] for x in representative_indexes]
    classifier = neural_network.MLPClassifier()
    classifier.fit(representative_inputs, representative_outputs)  # fit with the most representative data
    computed_outputs = classifier.predict(test_inputs)
    return computed_outputs
