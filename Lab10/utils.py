import pandas as pd


def load_data_spam(filename):
    file = pd.read_csv(filename)
    input_data = [value for value in file["emailText"]]
    output_data = [value for value in file["emailType"]]
    label_names = list(set(output_data))
    return input_data, output_data, label_names


def load_data_emotions(filename):
    file = pd.read_csv(filename)
    input_data = [value for value in file["Text"]]
    output_data = [value for value in file["Sentiment"]]
    label_names = list(set(output_data))
    return input_data, output_data, label_names
