import tensorflow as tf
from utility import *
from BGDRegression import *


def univariate_regression_by_tool(train_input, train_output):
    inputs = tf.keras.layers.Input(shape=(1,))
    outputs = tf.keras.layers.Dense(1, activation='linear')(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.SGD()
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(train_input, train_output, batch_size=len(train_input), epochs=300, verbose=1, shuffle=False)
    noOfPoints = 1000
    xref = []
    val = min(train_input)
    step = (max(train_input) - min(train_input)) / noOfPoints
    for i in range(1, noOfPoints):
        xref.append(val)
        val += step
    plotData(train_input, train_output, xref, model.predict(xref), [], [], title="train data and model with tool")


def univariate_regression(train_input, train_output, validation_input):
    data = [[el] for el in train_input]
    regressor = MyBGDRegression()
    regressor.fit(data, train_output)
    w0, w1 = regressor.intercept_, regressor.coef_[0]
    computed_outputs = regressor.predict([[x] for x in validation_input])
    return w0, w1, computed_outputs


def multivariate_regression(train_inputs, train_output, validation_inputs):
    data = [[el1, el2] for el1, el2 in zip(*train_inputs)]
    regressor = MyBGDRegression()
    regressor.fit(data, train_output)
    w0, w1, w2 = regressor.intercept_, regressor.coef_[0], regressor.coef_[1]
    computed_outputs = regressor.predict([[x, y] for x, y in zip(*validation_inputs)])
    return w0, w1, w2, computed_outputs