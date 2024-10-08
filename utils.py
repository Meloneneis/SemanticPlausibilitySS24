import os
import numpy as np
import matplotlib.pyplot as plt


def plot_accuracies(epoch_accuracies, title):
    # Plotting the accuracy per epoch for each seed
    plt.figure(figsize=(10, 6))
    all_accuracies = []

    for seed, accuracies in epoch_accuracies.items():
        epochs = range(1, len(accuracies) + 1)
        plt.plot(epochs, accuracies, marker='o', linestyle='-', label=f'Seed {seed}')
        all_accuracies.append(accuracies)

    # Calculate the average accuracy for each epoch
    avg_accuracies = np.mean(all_accuracies, axis=0)
    epochs = range(1, len(avg_accuracies) + 1)
    plt.plot(epochs, avg_accuracies, marker='x', linestyle='--', color='black', label='Average')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{os.path.join('plots', title)}.png")
    plt.show()


# Flip the dataset probabilties
def flip_function(example):
    example["p(NoHallucination)"] = abs(example['p(Hallucination)'] - 1.0)
    return example


# get the top and bottom n of the scores
def percentile_score(scores, n):
    scores = scores.tolist()
    scores = sorted(scores)
    top_n = scores[len(scores) - n]
    bottom_n = scores[n - 1]
    return top_n, bottom_n
