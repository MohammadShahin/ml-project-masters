from src.data_visualizer.kpa_visualizer import show_all_kp
from src.data_visualizer.ukp_visualizer import show_all_essays

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_visualizer.web_visualizer import show_all_web


def get_confusion():
    # Define the precision, recall, and actual number of labels for each class
    precision = [0.7289, 0.8667, 0.8912]
    recall = [0.8051, 0.8182, 0.8779]
    actual_labels = [354, 429, 868]

    # Calculate the true positives (TP), false positives (FP), and false negatives (FN) for each class
    TP = np.round(np.array(precision) * np.array(recall) * np.array(actual_labels))
    FP = np.round((1 - np.array(precision)) * np.array(actual_labels))
    FN = np.round((1 - np.array(recall)) * np.array(actual_labels))

    # Create a 2D array to store the confusion matrix values
    confusion_matrix = np.array([TP, FP, FN])

    # Optionally, normalize the confusion matrix
    # confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=0)

    # Print the confusion matrix
    return confusion_matrix


if __name__ == '__main__':
    show_all_kp()
    # f1 measure
    # data = {
    #     'SVM': [0.8504, 0.6675, 0.7137],
    #     'BERT': [0.7715, 0.6247, 0.5834],
    #     'Stack': [0.7880, 0.6363, 0.6404],
    #     'SVM-SVM': [0.8452, 0.6631, 0.6631],
    #     'BERT-BERT': [0.8424, 0.6478, 0.5876],
    #     'BERT-SVM': [0.8689, 0.7466, 0.6495],
    #     'SVM-BERT': [0.8845, 0.8417, 0.7651]
    # }
    # recall

    # data = {
    #     'SVM': [0.8764, 0.6497, 0.6798],
    #     'BERT': [0.7972, 0.6246, 0.5395],
    #     'Stack': [0.8329, 0.6177, 0.5734],
    #     'SVM-SVM': [0.8791, 0.6345, 0.6759],
    #     'BERT-BERT': [0.8624, 0.6078, 0.5325],
    #     'BERT-SVM': [0.8479, 0.8928, 0.5339],
    #     'SVM-BERT': [0.8779, 0.8182, 0.8051]
    # }
    #  precision
    # data = {
    #     'SVM': [0.8259, 0.6863, 0.7511],
    #     'BERT': [0.7454, 0.6212, 0.6523],
    #     'Stack': [0.7477, 0.6559, 0.7250],
    #     'SVM-SVM': [0.8138, 0.6944, 0.7500],
    #     'BERT-BERT': [0.8166, 0.6854, 0.6323],
    #     'BERT-SVM': [0.8910, 0.6415, 0.8289],
    #     'SVM-BERT': [0.8912, 0.8667, 0.7289]
    # }

    # data = {
    #     'SVM': [0.7439, 0.7733],
    #     'BERT': [0.6634, 0.6987],
    #     'Stack': [0.6882, 0.7169],
    #     'SVM-SVM': [0.7398, 0.7688],
    #     'BERT-BERT': [0.6926, 0.7237],
    #     'BERT-SVM': [0.7550, 0.7901],
    #     'SVM-BERT': [0.8304, 0.8478]
    # }
    # # Create x-axis values
    # x = np.arange(7)
    #
    # # Create figure and axis objects
    # fig, ax = plt.subplots()
    #
    # # Set width of each mini bar
    # width = 0.25
    #
    # # Plot the mini bars for each x-value
    # c_names = ['Macro AVG', 'Weighted', 'Non-Arguments']
    # r_names = ['SVM',
    #            'BERT',
    #            'Stack',
    #            'SVM-SVM',
    #            'BERT-BERT',
    #            'BERT-SVM',
    #            'SVM-BERT']
    # for i in range(2):
    #     ax.bar(x + i * width, [data[key][i] for key in data], width, label=c_names[i])
    #
    # # Set x-axis labels
    # ax.set_xticks(x)
    # plt.xticks(fontsize=8)
    # ax.set_xticklabels([r_names[i] for i in range(7)])
    #
    # # Set y-axis label
    # ax.set_ylabel('Value')
    # plt.title('Macro and Weighted AVG Values', fontsize=14)
    # # Add legend
    # ax.legend()
    #
    # # Show the plot
    # plt.show()

    # data = {
    #     'SVM': [0.8504, 0.6675, 0.7137],
    #     'BERT': [0.7715, 0.6247, 0.5834],
    #     'Stack': [0.7880, 0.6363, 0.6404],
    #     'SVM-SVM': [0.8452, 0.6631, 0.6631],
    #     'BERT-BERT': [0.8424, 0.6478, 0.5876],
    #     'BERT-SVM': [0.8689, 0.7466, 0.6495],
    #     'SVM-BERT': [0.8845, 0.8417, 0.7651]
    # }
    #
    # # Extract the values from the dictionary
    # values = [val for sublist in data.values() for val in sublist]
    #
    # # Plot the histogram
    # plt.hist(values, bins=10, edgecolor='black')
    #
    # # Set the labels and title
    # plt.xlabel('Values')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Values')
    #
    # # Show the plot
    # plt.show()

    # import matplotlib.pyplot as plt
    #
    # # Data
    # labels = ['SVM', 'BERT']
    # f1_measure = [0.67, 0.87]
    # precision = [0.68, 0.87]
    # recall = [0.65, 0.86]
    #
    # # Plotting
    # x = range(len(labels))
    # width = 0.1
    #
    # fig, ax = plt.subplots()
    # rects1 = ax.bar(x, f1_measure, width, label='F1-Measure')
    # rects2 = ax.bar([i + width for i in x], recall, width, label='Recall')
    # rects3 = ax.bar([i + 2 * width for i in x], precision, width, label='Precision')
    #
    # # Labeling
    # ax.set_ylabel('Scores')
    # ax.set_title('Negative Measurements Values')
    # ax.set_xticks([i + width for i in x])
    # ax.set_xticklabels(labels)
    # ax.legend()
    #
    # # Displaying the histogram
    # plt.show()
    # import numpy as np
    # import matplotlib.pyplot as plt

    # Define the class names
    # y_names = ['Premise', 'Claim', 'Non-Argument']
    # x_names = ['Non-Argument', 'Claim', 'Premise']
    #
    # # Define the confusion matrix values
    # confusion_matrix = get_confusion()
    #
    # # Create a figure and axis
    # fig, ax = plt.subplots()
    #
    # # Create the heatmap
    # heatmap = sns.heatmap(confusion_matrix, annot=True, cmap='YlGnBu', fmt='.0f', xticklabels=x_names,
    #                       yticklabels=y_names, ax=ax)
    #
    # # Set the axis labels
    # ax.set_xlabel('Predicted')
    # ax.set_ylabel('Actual')
    #
    # # Set the title
    # ax.set_title('Confusion Matrix for Model : SVM-BERT')
    #
    # # Rotate the x-axis tick labels
    # plt.xticks()
    #
    # # Display the heatmap
    # plt.show()
