import os
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from langdetect import detect


# function generated via GPT-4o
def get_top_words(ds, top_n=20):
    task_names = set(ds[list(ds.keys())[0]]["task"])
    word_freq_dict = {task_name: Counter() for task_name in task_names}

    for task, src_text in zip(ds[list(ds.keys())[0]]["task"], ds[list(ds.keys())[0]]["src"]):
        words = src_text.split()
        word_freq_dict[task].update(words)

    top_words_dict = {task: word_freq_dict[task].most_common(top_n) for task in task_names}
    return top_words_dict


def plot_word_frequencies(word_freq_dict, title):
    fig, axs = plt.subplots(len(word_freq_dict), 1, figsize=(10, 5 * len(word_freq_dict)))
    if len(word_freq_dict) == 1:
        axs = [axs]  # Ensure axs is iterable when there's only one subplot

    for ax, (task, word_freq) in zip(axs, word_freq_dict.items()):
        words, freqs = zip(*word_freq)
        ax.bar(words, freqs)
        ax.set_title(f"Most frequent Words for Task: {task}")
        ax.set_xlabel("Words")
        ax.set_ylabel("Frequency")
        ax.tick_params(axis='x', rotation=90)

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def get_task_count(ds):
    task_names = set(ds[list(ds.keys())[0]]["task"])
    count_dict = {}
    for task_name in task_names:
        # if a task name element in the dataset equals the task name then add 1 to the count
        count_dict[f"{task_name}_count"] = sum(1 for element in ds[list(ds.keys())[0]]["task"] if task_name == element)
    return count_dict


def get_lang_count(ds):
    lang_names = set(language for language, _ in ds)
    count_dict = {}
    for lang in lang_names:
        count_dict[f"{lang}_count"] = sum(1 for lang_name, _ in ds if lang == lang_name)
    return count_dict


def get_hallucination_probs(ds):
    task_names = set(ds[list(ds.keys())[0]]["task"])
    length_dict = {task_name: [] for task_name in task_names}

    for task, hallu_prob in zip(ds[list(ds.keys())[0]]["task"], ds[list(ds.keys())[0]]["p(Hallucination)"]):
        length_dict[task].append(hallu_prob)

    return length_dict


def get_task_and_length(ds, length_type='encoding'):
    task_names = set(ds[list(ds.keys())[0]]["task"])
    length_dict = {task_name: [] for task_name in task_names}

    for task, src_text in zip(ds[list(ds.keys())[0]]["task"], ds[list(ds.keys())[0]]["src"]):
        if length_type == 'encoding':
            length = len(tokenizer(src_text, add_special_tokens=False)["input_ids"])
        elif length_type == 'words':
            length = len(src_text.split())
        else:
            raise ValueError("Invalid length_type. Please use 'encoding' or 'words'.")
        length_dict[task].append(length)

    return length_dict


def show_hist(datas,title, n_bins=None):
    fig, axs = plt.subplots(1, len(datas), sharey=True, tight_layout=True)
    for i, (task, encoding_lengths) in enumerate(datas.items()):
        if n_bins is None:
            unique_values = len(set(encoding_lengths))
            axs[i].hist(encoding_lengths, bins=unique_values)
        else:
            axs[i].hist(encoding_lengths, bins=n_bins)
        axs[i].set_title(f"Task {task}")
    fig.suptitle(title)
    plt.show()



# loading the csv dataset
train_ds = load_dataset("json", data_files={
    "train": os.path.join("shroom_ds", "SHROOM_unlabeled-training-data-v2", "train.model-agnostic.json")})
valid_ds = load_dataset("json", data_files={
    "valid": os.path.join("shroom_ds", "SHROOM_dev-v2", "val.model-agnostic.json")})
test_ds = load_dataset("json", data_files={
    "test": os.path.join("shroom_ds", "SHROOM_test-labeled", "test.model-agnostic.json")})
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

print("Starting Dataset Analysis...")
print(f"Number of training examples: {len(train_ds['train'])}")
print(f"Number of validation examples: {len(valid_ds['valid'])}")
print(f"Number of test examples: {len(test_ds['test'])}")
print(f"Count of each task in train set: {get_task_count(train_ds)}")
print(f"Count of each task in valid set: {get_task_count(valid_ds)}")
print(f"Count of each task in test set: {get_task_count(test_ds)}")
print("Now let's plot the token length distribution wrt. to the task and tokenizer (roberta-base)...")
train_encoding_lengths = get_task_and_length(train_ds, "encoding")
valid_encoding_lengths = get_task_and_length(valid_ds, "encoding")
test_encoding_lengths = get_task_and_length(test_ds, "encoding")
show_hist(train_encoding_lengths, n_bins=20, title="Token Length Distribution for Train Set")
show_hist(valid_encoding_lengths, n_bins=20, title="Token Length Distribution for Valid Set")
show_hist(valid_encoding_lengths, n_bins=20, title="Token Length Distribution for Test Set")
print("Next, let's plot how many tokens were used to encode a word...")
train_word_count = get_task_and_length(train_ds, "words")
valid_word_count = get_task_and_length(valid_ds, "words")
test_word_count = get_task_and_length(test_ds, length_type="words")
show_hist(train_word_count, n_bins=20, title="Word Count Distribution for Train Set")
show_hist(valid_word_count, n_bins=20, title="Word Count Distribution for Valid Set")
show_hist(test_word_count, n_bins=20, title="Word Count Distribution for Test Set")
print("Next, let's plot the top 20 most frequent words for each task")
train_top_words = get_top_words(train_ds, 50)
valid_top_words = get_top_words(valid_ds, 50)
test_top_words = get_top_words(test_ds, 50)
plot_word_frequencies(train_top_words, title="Top 50 Words in Training Set")
plot_word_frequencies(valid_top_words, title="Top 50 Words in Validation Set")
plot_word_frequencies(test_top_words, title="Top 50 Words in Test Set")
print("Given the word frequencies, it seems to be that the task MT has a lot of russian characters. \n"
      "Let's check the languages of the documents in the MT task next..")
valid_languages = [(detect(document["src"]), document["src"]) for document in valid_ds["valid"] if document["task"] == "MT"]
test_languages = [(detect(document["src"]), document["src"]) for document in test_ds["test"] if document["task"] == "MT"]
valid_lang_count = get_lang_count(valid_languages)
test_lang_count = get_lang_count(test_languages)
print(f"Validation Set: {valid_lang_count}")
print(f"Test Set: {test_lang_count}")
print("Given the lang count results, most documents are in russian and there are a few which are not. \n"
      "This could potentially be due to incorrect language detection. However, all languages are slavic languages.")
print("Lastly let's plot the hallucination distribution wrt task")
val_hallucination_count = get_hallucination_probs(valid_ds)
test_hallucination_count = get_hallucination_probs(test_ds)
show_hist(val_hallucination_count, "Hallucination Distribution for Validation Set")
show_hist(test_hallucination_count, "Hallucination Distribution for Test Set")


