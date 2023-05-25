import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from predictions import predict, unsupervised_predict, supervised_predict, hybrid_predict
from utils import *


def split_data(inputs, outputs):
    # np.random.seed(4)
    indexes = [i for i in range(len(inputs))]
    train_sample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    test_sample = [i for i in indexes if i not in train_sample]
    train_inputs = [inputs[i] for i in train_sample]
    train_outputs = [outputs[i] for i in train_sample]
    test_inputs = [inputs[i] for i in test_sample]
    test_outputs = [outputs[i] for i in test_sample]
    return train_inputs, train_outputs, test_inputs, test_outputs


def extract_features_with_tfidf(train_inputs, test_inputs, max_features):
    tfidf = TfidfVectorizer(max_features=max_features)
    train_features = tfidf.fit_transform(train_inputs)
    test_features = tfidf.fit_transform(test_inputs)
    return train_features.toarray(), test_features.toarray()


def extract_features_with_one_hot_encoding(train_inputs, test_inputs):
    ohe = OneHotEncoder()
    train_features = ohe.fit_transform(train_inputs)
    test_features = ohe.transform(test_inputs)
    return train_features.toarray(), test_features.toarray()


def extract_features_with_ngrams(train_inputs, test_inputs):
    cv = CountVectorizer(ngram_range=(1, 2))
    train_features = cv.fit_transform(train_inputs)
    test_features = cv.transform(test_inputs)
    return train_features.toarray(), test_features.toarray()


def run_spam():
    # READ DATA
    inputs, outputs, output_names = load_data_spam("data/spam.csv")

    # SPLIT DATA
    train_inputs, train_outputs, test_inputs, test_outputs = split_data(inputs, outputs)

    # CLASSIFICATION USING KMEANS
    print("\n####################### SPAM: #####################\n")
    train_features, test_features = extract_features_with_ngrams(train_inputs, test_inputs)

    computed_output, centroids, computed_indexes = predict(train_features, test_features, output_names, len(set(output_names)))
    inverseTestOutputs = ['spam' if elem == 'ham' else 'ham' for elem in test_outputs]
    acc = accuracy_score(test_outputs, computed_output)
    acc_2 = accuracy_score(inverseTestOutputs, computed_output)

    print("Real output: ", test_outputs)
    print("My output:   ", computed_output)
    print("Accuracy: ", max(acc, acc_2))
    print("\n###################################################\n")


def run_emotions():
    # READ DATA
    inputs, outputs, output_names = load_data_emotions("data/reviews_mixed.csv")

    # SPLIT DATA
    train_inputs, train_outputs, test_inputs, test_outputs = split_data(inputs, outputs)

    print("\n##################### EMOTIONS: ###################\n")
    trainFeatures, testFeatures = extract_features_with_tfidf(train_inputs, test_inputs, 150)

    unsupervised_output = unsupervised_predict(trainFeatures, testFeatures, output_names, len(set(output_names)))
    supervised_output = supervised_predict(trainFeatures, train_outputs, testFeatures)
    hybrid_output = hybrid_predict(trainFeatures, train_outputs, testFeatures, test_outputs)
    inverseTestOutputs = ['negative' if elem == 'positive' else 'positive' for elem in test_outputs]

    accuracyByTool = accuracy_score(test_outputs, unsupervised_output)
    accuracyByToolInverse = accuracy_score(inverseTestOutputs, unsupervised_output)

    print("Unsupervised output: ", unsupervised_output)
    print("Supervised output:   ", supervised_output)
    print("Hybrid output:       ", hybrid_output)
    print("Real output:         ", test_outputs)

    print("Accuracy unsupervised: ", max(accuracyByTool, accuracyByToolInverse))
    print("Accuracy supervised: ", accuracy_score(test_outputs, supervised_output))
    print("Accuracy hybrid: ", accuracy_score(test_outputs, hybrid_output))
    print("\n###################################################\n")


if __name__ == '__main__':
    run_spam()
    run_emotions()
