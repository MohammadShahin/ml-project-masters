import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Read the JSON file
data_dir = os.path.abspath(os.getcwd()) + '/corpora/parsed-corpora/'

data = pd.read_json(data_dir + 'args_sentences.json')
data = data[data['train']]

arg_score_mace_p = data['arg-score-mace-p']
arg_score_wa = data['arg-score-wa']
topics = data['arg-topic']
grouped_data = data.groupby('arg-topic')
mean_scores = data.groupby('arg-topic')['arg-score-mace-p'].mean()
filtered_data = data[data['arg-topic'].isin(mean_scores[mean_scores <= 0.6].index)]

print('topics : ', len(topics))


def show_macep_score_ibm():
    plt.hist(arg_score_mace_p, bins=30, edgecolor='black')
    plt.xlabel('MACE-P Score')
    plt.ylabel('Frequency')
    plt.title('MACE-P Score Distribution')
    plt.show()


def show_wa_score_ibm():
    plt.hist(mean_scores, bins=40, edgecolor='black')
    plt.xlabel('Mean Scores')
    plt.ylabel('Frequency')
    plt.title('arg-score-wa Distribution')
    plt.show()


def show_topic_ibm():
    plt.hist(mean_scores, bins=30, edgecolor='black')
    plt.xlabel('MACE-P Score Mean')
    plt.ylabel('Frequency')
    plt.title('MACE-P Score Mean Distribution Over Topics')
    plt.show()


def show_topic_scores():
    # Plot the mean scores
    mean_scores.plot(kind='bar')

    # Set labels and title
    plt.xlabel('Topic')
    plt.ylabel('Mean Score')
    plt.title('Mean Scores by Topic')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.show()


def show_all_ibm():
    show_macep_score_ibm()
    show_topic_ibm()
    # show_topic_scores()
