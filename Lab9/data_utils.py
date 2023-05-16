import numpy as np


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


def evaluate(real_labels, computed_labels, label_names):
    from sklearn.metrics import confusion_matrix

    conf_matrix = confusion_matrix(real_labels, computed_labels)
    acc = sum([conf_matrix[i][i] for i in range(len(label_names))]) / len(real_labels)
    precision = {}
    recall = {}
    for i in range(len(label_names)):
        precision[label_names[i]] = conf_matrix[i][i] / sum([conf_matrix[j][i] for j in range(len(label_names))])
        recall[label_names[i]] = conf_matrix[i][i] / sum([conf_matrix[i][j] for j in range(len(label_names))])
    return acc, precision, recall, conf_matrix
