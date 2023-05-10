from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import numpy as np
from utils import *
from MulticlassRegression import MyMulticlassLogisticRegression
from sklearn.linear_model import SGDClassifier


def split_data(inputs, outputs):
    np.random.seed(4)
    indexes = [i for i in range(len(inputs))]
    train_sample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    test_sample = [i for i in indexes if i not in train_sample]
    train_inputs = [inputs[i] for i in train_sample]
    train_outputs = [outputs[i] for i in train_sample]
    test_inputs = [inputs[i] for i in test_sample]
    test_outputs = [outputs[i] for i in test_sample]
    return train_inputs, train_outputs, test_inputs, test_outputs


def normalise_data(train_data, test_data):
    scaler = StandardScaler()
    scaler.fit(train_data)
    normalisedTrainData = scaler.transform(train_data)
    normalisedTestData = scaler.transform(test_data)
    return normalisedTrainData, normalisedTestData


def calculate_error(computed_outputs, test_outputs):
    error = 0.0
    for t1, t2 in zip(computed_outputs, test_outputs):
        if t1 != t2:
            error += 1
    error = error / len(test_outputs)
    print("Error: ", error)


def multiclass_classification_by_tool(train_inputs, train_outputs, test_inputs, test_outputs):
    regressor = linear_model.LogisticRegression()
    regressor.fit(train_inputs, train_outputs)

    print_regression_models(regressor)

    computed_outputs = regressor.predict(test_inputs)
    print("Accuracy: ", regressor.score(test_inputs, test_outputs))
    return computed_outputs


def multiclass_classification_by_me(train_inputs, train_outputs, test_inputs, test_outputs):
    regressor = MyMulticlassLogisticRegression()
    regressor.fit(train_inputs, train_outputs)

    print_regression_models(regressor)

    computed_outputs = regressor.predict(test_inputs)

    accuracy = 0.0
    for i in range(len(test_inputs)):
        if test_outputs[i] == computed_outputs[i]:
            accuracy += 1
    print("Accuracy: ", accuracy / len(test_inputs))
    return computed_outputs


def other_loss_function(train_inputs, train_outputs, test_inputs, test_outputs):
    regressor = SGDClassifier(loss='log_loss')
    regressor.fit(train_inputs, train_outputs)
    print("Accuracy with LOG LOSS:", regressor.score(test_inputs, test_outputs))

    regressor = SGDClassifier(loss='hinge')
    regressor.fit(train_inputs, train_outputs)
    print("Accuracy with HINGE LOSS:", regressor.score(test_inputs, test_outputs))

    regressor = SGDClassifier(loss='squared_hinge')
    regressor.fit(train_inputs, train_outputs)
    print("Accuracy with SQUARED HINGE LOSS:", regressor.score(test_inputs, test_outputs))


def run():
    # READ DATA
    inputs, outputs, outputNames, feature1, feature2, feature3, feature4, featureNames = load_data_flowers()
    # plot_data_four_features(inputs, outputs, outputNames, featureNames, "Initial data for flowers")

    # PLOT HISTOGRAMS
    plot_histogram_feature(feature1, featureNames[0])
    plot_histogram_feature(feature2, featureNames[1])
    plot_histogram_feature(feature3, featureNames[2])
    plot_histogram_feature(feature4, featureNames[3])
    plot_histogram_feature(outputs, 'Flowers class')

    # SPLIT AND NORMALISE DATA
    train_inputs, train_outputs, test_inputs, test_outputs = split_data(inputs, outputs)
    train_inputs, test_inputs = normalise_data(train_inputs, test_inputs)
    # plot_data_four_features(train_inputs, train_outputs, outputNames, featureNames, "Normalised flowers' data")

    # CLASSIFICATION USING TOOL
    print("\n########## Classification model by TOOL: ##########\n")
    computed_outputs = multiclass_classification_by_tool(train_inputs, train_outputs, test_inputs, test_outputs)
    # plot_predictions(test_inputs, test_outputs, computed_outputs, outputNames, featureNames[:2], "Results by tool")
    calculate_error(computed_outputs, test_outputs)
    print("\n###################################################\n")

    # CLASSIFICATION USING CODE
    print("\n########### Classification model by ME: ###########\n")
    computed_outputs = multiclass_classification_by_me(train_inputs, train_outputs, test_inputs, test_outputs)
    # plot_predictions(test_inputs, test_outputs, computed_outputs, outputNames, featureNames[:2], "Results by me")
    calculate_error(computed_outputs, test_outputs)
    print("\n###################################################\n")

    # LOSS FUNCTIONS
    print("\n################# Loss functions: #################\n")
    train_inputs, train_outputs, test_inputs, test_outputs = split_data(inputs, outputs)
    train_inputs, test_inputs = normalise_data(train_inputs, test_inputs)
    other_loss_function(train_inputs, train_outputs, test_inputs, test_outputs)
    print("\n###################################################\n")


if __name__ == '__main__':
    run()
