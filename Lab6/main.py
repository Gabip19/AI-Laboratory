import numpy as np
from sklearn import linear_model

from custom_regression import CustomLiniarBivariateRegression
from utils import load_data, plot_train_and_validation, plot_data_histogram, plot_full_data


def train_and_test(attributes, result):
    np.random.seed(5)
    indexes = [i for i in range(len(result))]
    train_sample_indexes = np.random.choice(indexes, int(0.8 * len(result)), replace=False)

    validation_sample_indexes = [i for i in range(len(result)) if i not in train_sample_indexes]

    train_values = [attributes[i] for i in train_sample_indexes]
    train_result = [result[i] for i in train_sample_indexes]

    validation_values = [attributes[i] for i in validation_sample_indexes]
    validation_result = [result[i] for i in validation_sample_indexes]

    return train_values, train_result, validation_values, validation_result


def linear_regression_by_tool(train_attributes, train_result):
    regressor_result = linear_model.LinearRegression()
    regressor_result.fit(train_attributes, train_result)
    return regressor_result


def liniar_regression(train_attributes, train_result):
    regressor_result = CustomLiniarBivariateRegression()
    regressor_result.fit(train_attributes, train_result)
    return regressor_result


def mean_absolute_error(computed_output, validation_output):
    error = 0.0
    for t1, t2 in zip(computed_output, validation_output):
        error += (t1 - t2) ** 2
    error = error / len(validation_output)
    print('Prediction error: ', error)


if __name__ == '__main__':
    inputs, outputs = load_data(
        'data/v1_world-happiness-report-2017.csv',
        ['Economy..GDP.per.Capita.', 'Freedom'],
        'Happiness.Score'
    )

    plot_data_histogram([value[0] for value in inputs], 'GDP')
    plot_data_histogram([value[1] for value in inputs], 'Freedom')
    plot_data_histogram(outputs, 'Happiness score')

    train_inputs, train_outputs, validation_inputs, validation_outputs = train_and_test(inputs, outputs)
    plot_train_and_validation(train_inputs, train_outputs, validation_inputs, validation_outputs)

    regressor_ = linear_regression_by_tool(train_inputs, train_outputs)
    w = [regressor_.intercept_, regressor_.coef_[0], regressor_.coef_[1]]
    print('The learnt model by tool: f(x) = ', w[0], ' + ', w[1], ' * x1  +', w[2], ' * x2')
    predicted_outputs = regressor_.predict(validation_inputs)
    # mean_absolute_error(predicted_outputs, validation_outputs)

    regressor = liniar_regression(train_inputs, train_outputs)
    w = regressor.w
    print('The learnt model by me: f(x) = ', w[0], ' + ', w[1], ' * x1  +', w[2], ' * x2')
    predicted_outputs = regressor.predict(validation_inputs)
    mean_absolute_error(predicted_outputs, validation_outputs)

    plot_full_data([train_inputs, train_outputs], [validation_inputs, validation_outputs])
