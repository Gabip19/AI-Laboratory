from AbstractClassifier import AbstractClassifier
from CNNSepiaClassifier import CNNSepiaClassifier
from utils import *
from sklearn.preprocessing import StandardScaler
from NeuralClassifier import *
from ANNSepiaClassifier import ANNSepiaClassifier
from data_utils import *


def normalise_data(train_data, test_data):
    scaler = StandardScaler()
    scaler.fit(train_data)
    normalised_train_data = scaler.transform(train_data)
    normalised_test_data = scaler.transform(test_data)
    return normalised_train_data, normalised_test_data


def neural_classification(train_inputs, train_outputs, test_inputs):
    classifier = NeuralClassifier()

    classifier.fit(np.array(train_inputs), np.array(train_outputs))

    computed_outputs = classifier.predict(test_inputs)
    return computed_outputs


def flatten(mat):
    x = []
    for line in mat:
        for el in line:
            x.append(el)
    return x


def flatten_data(train_inputs, test_inputs):
    train_inputs = [flatten(el) for el in train_inputs]
    test_inputs = [flatten(el) for el in test_inputs]
    return train_inputs, test_inputs


def run_flowers():
    # READ DATA
    inputs, outputs, output_names, feature1, feature2, feature3, feature4, feature_names = load_flowers_data()

    # PLOT HISTOGRAMS
    plot_histogram_feature(feature1, feature_names[0])
    plot_histogram_feature(feature2, feature_names[1])
    plot_histogram_feature(feature3, feature_names[2])
    plot_histogram_feature(feature4, feature_names[3])
    plot_histogram_feature(outputs, "Flowers")

    # SPLIT AND NORMALISE DATA
    train_inputs, train_outputs, test_inputs, test_outputs = split_data(inputs, outputs)
    train_inputs, test_inputs = normalise_data(train_inputs, test_inputs)

    # CLASSIFICATION USING CODE
    print("\n##################### Flowers: ####################\n")
    computed_outputs = neural_classification(train_inputs, train_outputs, test_inputs)

    acc, precision, recall, conf_matrix = evaluate(test_outputs, computed_outputs, output_names)
    print("Accuracy: ", acc)
    print("Precision: ", precision)
    print("Recall: ", recall)
    plot_confusion_matrix(conf_matrix, output_names, "Iris flowers classification")
    print("\n###################################################\n")


def run_digits():
    # READ DATA
    inputs, outputs, output_names = load_digits_data()

    # PLOT HISTOGRAMS
    plot_histogram_data(outputs, output_names, "Digits")

    # SPLIT AND NORMALISE DATA
    train_inputs, train_outputs, test_inputs, test_outputs = split_data(inputs, outputs)
    train_inputs, test_inputs = flatten_data(train_inputs, test_inputs)
    train_inputs, test_inputs = normalise_data(train_inputs, test_inputs)

    # CLASSIFICATION USING CODE
    print("\n##################### Digits: #####################\n")
    computed_outputs = neural_classification(train_inputs, train_outputs, test_inputs)

    acc, precision, recall, conf_matrix = evaluate(test_outputs, computed_outputs, output_names)
    print("Accuracy: ", acc)
    print("Precision: ", precision)
    print("Recall: ", recall)
    plot_confusion_matrix(conf_matrix, output_names, "Digits classification")
    print("\n###################################################\n")


def run():
    run_flowers()
    run_digits()

    input_data, output_data = AbstractClassifier.load_data()
    train_inputs, train_outputs, test_inputs, test_outputs = split_data(input_data, output_data)

    print("\n###################### ANN: #######################\n")
    ann_classifier = ANNSepiaClassifier()
    ann_classifier.run_classifier(train_inputs, train_outputs, test_inputs, test_outputs)
    print("\n###################################################\n")

    print("\n###################### CNN: #######################\n")
    cnn_classifier = CNNSepiaClassifier()
    cnn_classifier.run_classifier(train_inputs, train_outputs, test_inputs, test_outputs)
    print("\n###################################################\n")


if __name__ == '__main__':
    run()
