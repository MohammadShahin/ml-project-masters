import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

# Read the JSON file
data_dir = os.path.abspath(os.getcwd()) + '/corpora/parsed-corpora/'

data = pd.read_json(data_dir + 'kp_data.json')
args = data['arg_id']
arg_freq = Counter(args)
arg_freq = Counter(arg_freq.values())
kp = data['key_point_id']
kp_freq = Counter(kp)
kp_freq = Counter(kp_freq.values())
label = data['label']


def show_arguments():
    integers = list(arg_freq.keys())
    counts = list(arg_freq.values())
    # Plot the histogram
    plt.bar(integers, counts)
    plt.xlabel('Argument Frequency')
    plt.ylabel('Number of Arguments')
    plt.title('Arguments Frequency Histogram')
    plt.show()


def show_keypoints():
    integers = list(kp_freq.keys())
    counts = list(kp_freq.values())
    # Plot the histogram
    plt.bar(integers, counts)
    plt.xlabel('Key-Points Frequency')
    plt.ylabel('Number of Key-Points')
    plt.title('Key-Points Frequency Histogram')
    plt.show()


def show_labels():
    plt.hist(label, bins=4, edgecolor='black')
    plt.xticks([0, +1], ['Non-Matching', 'Matching'])
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Histogram of Matching Class')
    plt.show()


def show_all_kp():
    show_arguments()
    show_keypoints()
    show_labels()
    # print(len(data))
