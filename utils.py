import os
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification

def plot_accuracies(epoch_accuracies, title):
    # Plotting the accuracy per epoch for each seed
    plt.figure(figsize=(10, 6))
    for seed, accuracies in epoch_accuracies.items():
        epochs = range(1, len(accuracies) + 1)
        plt.plot(epochs, accuracies, marker='o', linestyle='-', label=f'Seed {seed}')

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




def model_init():
    return AutoModelForSequenceClassification.from_pretrained('OpenAssistant/reward-model-deberta-v3-base',
                                                              num_labels=1,
                                                              output_attentions=False,
                                                              output_hidden_states=False,
                                                              return_dict=True
                                                              )


# this function was created with GPT-4o
def percentile_score(scores, n):
    scores = scores.tolist()
    scores = sorted(scores)
    top_n = scores[len(scores)-n]
    bottom_n = scores[n-1]

    return top_n, bottom_n

