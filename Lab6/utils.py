import pandas as pd
import matplotlib.pyplot as plt


def plot_data_histogram(x, variable_name):
    plt.hist(x, 10)
    plt.title('Histogram of ' + variable_name)
    plt.show()


def plot_train_and_validation(_train_inputs, _train_outputs, _validation_inputs, _validation_outputs):
    plt.plot(_train_inputs, _train_outputs, 'ro', label='training data')
    plt.plot(_validation_inputs, _validation_outputs, 'g^', label='validation data')
    plt.title('train and validation data')
    plt.xlabel('GDP and Freedom')
    plt.ylabel('Happiness')
    plt.legend()
    plt.show()


def plot_full_data(train, validation):
    ax = plt.axes(projection='3d')
    ax.plot3D([tr[0] for tr in train[0]], [tr[1] for tr in train[0]], train[1], 'ro', label='training data')
    ax.plot3D([val[0] for val in validation[0]], [val[1] for val in validation[0]], validation[1], 'g^',
              label='validation data')
    plt.legend()
    plt.show()


def load_data(filename, input_features, output_feature):
    file = pd.read_csv(filename)
    in_features = []
    out_feature = []
    for line_index in range(file.shape[0]):
        values_zip = []
        ok = True
        for feature in input_features:
            elem = file[feature][line_index]
            if pd.isna(elem):
                ok = False
            else:
                values_zip.append(float(elem))
        if ok:
            in_features.append(values_zip)
            out_feature.append(float(file[output_feature][line_index]))
    return in_features, out_feature
