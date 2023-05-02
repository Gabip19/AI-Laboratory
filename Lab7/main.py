from singletarget import *
from multitarget import *


def normalise_aux(input_list):
    mean_ = sum(input_list) / len(input_list)
    std = (1 / len(input_list) * sum([(i - mean_) ** 2 for i in input_list])) ** 0.5
    normalised_input = [(i - mean_) / std for i in input_list]
    return normalised_input


def normalise_data(data):
    normalised_data = []
    for d in data:
        normalised_inputs = normalise_aux(d)
        normalised_data.append(normalised_inputs)
    return normalised_data


def split_data(inputs, output):
    np.random.seed(5)
    indexes = [i for i in range(len(output))]
    train_sample = np.random.choice(indexes, int(0.8 * len(output)), replace=False)
    validation_sample = [i for i in indexes if i not in train_sample]
    train_inputs = []
    validation_inputs = []
    for inp in inputs:
        train_inputs.append([inp[i] for i in train_sample])
        validation_inputs.append([inp[i] for i in validation_sample])
    train_output = [output[i] for i in train_sample]
    validation_output = [output[i] for i in validation_sample]
    return train_inputs, train_output, validation_inputs, validation_output


def get_error(computed_output, validation_output):
    error = 0.0
    for r1, r2 in zip(computed_output, validation_output):
        error += (r1 - r2) ** 2
    return error / len(validation_output)


def run():
    filename = "data/v1_world-happiness-report-2017.csv"
    inputs, output = read_from_csv(filename, ["Economy..GDP.per.Capita.", "Freedom"], "Happiness.Score")
    inputs, output = prepare_data(inputs, output)
    data = normalise_data([*inputs, output])
    inputs, output = data[:2], data[2]
    train_inputs, train_output, validation_inputs, validation_output = split_data(inputs, output)

    print("Univariate regression with tool: ")
    univariate_regression_by_tool(train_inputs[0], train_output)
    print("\nUnivariate regression: ")
    w0, w1, result = univariate_regression(train_inputs[0], train_output, validation_inputs[0])
    print("My learnt model: f(x) = ", w0, " + ", w1, " * x")
    print("Prediction error: ", get_error(result, validation_output))

    print("\nMultivariate regression: ")
    train_inputs, train_output, validation_inputs, validation_output = split_data(inputs, output)
    w0, w1, w2, result = multivariate_regression(train_inputs, train_output, validation_inputs)
    print("My learnt model: f(x) = ", w0, " + ", w1, " * x1 + ", w2, " * x2")
    print("Prediction error: ", get_error(result, validation_output))
    plot_results(train_inputs, train_output, w0, w1, w2)

    inputs, outputs = read_from_csv_multitarget(filename, ["Economy..GDP.per.Capita.", "Freedom"],
                                                ["Happiness.Score", "Generosity"])
    inputs, outputs = prepare_data(inputs, outputs)
    data = normalise_data([*inputs, *outputs])
    inputs, outputs = data[:2], data[:-1]
    split_data_multitarget(inputs, outputs)
    print("\nMultitarget regression: ")
    results = multitarget_regression()
    print("Prediction error: ", get_error(results[-1], results[-2]))


if __name__ == '__main__':
    run()
