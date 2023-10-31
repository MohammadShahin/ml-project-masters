import matplotlib.pyplot as plt

import seaborn as sns
from scipy.stats import pearsonr

def TrainingValidationLoss(train_df, val_df):
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    # Plot the learning curve.
    plt.plot(train_df, 'b-o', label="Training")
    plt.plot(val_df, 'g-o', label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])

    plt.show()


def ValidationLoss(val_df):
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(val_df, 'g-o', label="Testing")

    # Label the plot.
    plt.title("Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])

    plt.show()


def PearsonMeasure(val_df, label_df):
    epoch = 1
    corr_df = []
    p_df = []
    for pred, label in zip(val_df, label_df):
        corr, p_value = pearsonr(pred, label)
        print("Epoch : ", epoch)
        epoch = epoch + 1
        print("Pearson correlation coefficient:", corr)
        corr_df.append(corr)
        print("p-value:", p_value)
        p_df.append(p_value)


    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(corr_df, 'b-o', label="Correlation")
    plt.plot(p_df, 'g-o', label="P-Value")

    # Label the plot.
    plt.title("Correlation & P-Value of Pearson (p) ")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.xticks([1, 2, 3, 4, 5])

    plt.show()
