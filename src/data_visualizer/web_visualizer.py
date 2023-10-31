import os
import matplotlib.pyplot as plt
import pandas as pd

# Read the JSON file
data_dir = os.path.abspath(os.getcwd()) + '/corpora/parsed-corpora/'

data = pd.read_json(data_dir + 'web_discourse.json')
arg_sent_len = data['sent-text'].apply(len)
arg_class = data['sent-class']


def show_sent_essays():
    plt.hist(arg_sent_len, bins=30, edgecolor='black')
    plt.xlabel('Sentence Length')
    plt.ylabel('Frequency')
    plt.title('Histogram of Sentence Length')
    plt.show()


def show_sent_class():
    plt.hist(arg_class, bins=5, edgecolor='black')
    plt.xticks(['n', 'c', 'p'], ['Non-Argument', 'Claim', 'Premise'])
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Histogram of Sentence Class')
    plt.show()


def show_all_web():
    show_sent_essays()
    show_sent_class()
    print(len(data))
