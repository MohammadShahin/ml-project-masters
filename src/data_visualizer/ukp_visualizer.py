import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

# Read the JSON file
data_dir = os.path.abspath(os.getcwd()) + '/corpora/parsed-corpora/'

data = pd.read_json(data_dir + 'essays_sentences.json')
arg_sent_len = data['sent-text'].apply(len)
arg_class = data['sent-class']
essays = data['essay-id']

max_paragraph_ids = data.groupby('essay-id')['parag-idx'].max() + 1
paragraphs = max_paragraph_ids.value_counts()
# print(paragraphs)


def show_sent_essays():
    plt.hist(arg_sent_len, bins=50, edgecolor='black')
    plt.xlabel('Sentence')
    plt.ylabel('Length')
    plt.title('Histogram of Sentences\' Length')
    plt.show()


def show_sent_class():

    plt.hist(arg_class, bins=5, edgecolor='black')
    plt.xticks(['n', 'c', 'p'], ['Non-Argument', 'Claim', 'Premise'])
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Histogram of Sentence Classes')
    plt.show()


def show_essays():
    # plt.hist(essays, bins=40, edgecolor='black')
    frequency = Counter(essays)
    print(len(frequency))
    x = list(range(len(frequency)))
    y = list(frequency.values())
    plt.bar(x, y)
    plt.xlabel('Essays')
    plt.ylabel('Sentences')
    plt.title('Histogram of Sentences in the Essays')
    plt.show()


def show_paragraphs():
    # plt.hist(paragraphs, bins=10, edgecolor='black')
    # plt.xlabel('Paragraphs')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Paragraphs\' Number per Essay')
    # plt.show()
    paragraphs.plot(kind='bar')

    # Set the labels and title
    plt.xlabel('Paragraphs Count')
    plt.ylabel('Number of Essays')
    plt.title('Histogram of Paragraphs\' Number per Essay')

    # Show the plot
    plt.show()


def show_all_essays():
    show_sent_essays()
    show_sent_class()
    show_essays()
    show_paragraphs()
