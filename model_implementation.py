import os
import torch
import json
import tqdm
from datasets import load_dataset, interleave_datasets, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding, TrainerCallback
import evaluate
from utils import flip_function, plot_accuracies
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy import stats
from peft import LoraConfig, get_peft_model, TaskType
"""
Note: I am aware that there is also the model-aware dataset, but for the sake of working with data scarcity I will
only consider the model-agnostic dataset.

As shown in the dataset analysis, we only have 499 labeled data points from the dev set and 30k unlabeled train data. 
As our task is it to train a classifier that predicts whether the input text has hallucination or not, we can do so
by using the cross-entropy loss function. However, this loss function requires labels so only the dev set data can be
used as is for training.

Hence, I will use 80% labeled dev set data as the training data and the remaining 20% labeled dev set data remains as is.
For the split I will make sure that the tasks has the original task ratios (which are somewhat balanced already).
"""
#################### Run Parameters ####################
skip_synthesizing_labels = False
preload_significance_values = True
skip_training_classifier = True

model_checkpoint = "OpenAssistant/reward-model-deberta-v3-large-v2"
########################################################
# create results directory for the epoch accuracies
if not os.path.exists("results"):
    os.makedirs("results")

# create plots directory for the plots
if not os.path.exists("plots"):
    os.makedirs("plots")

# create plots directory for the plots
if not os.path.exists("models"):
    os.makedirs("models")
# Load the dev set data
ds = load_dataset("json", data_files={"valid": os.path.join("shroom_ds", "SHROOM_dev-v2", "val.model-agnostic.json")})

# split the data into their respective tasks
ds_mt = ds.filter(lambda x: x["task"] == "MT")
ds_pg = ds.filter(lambda x: x["task"] == "PG")
ds_dm = ds.filter(lambda x: x["task"] == "DM")

# take the first 80% of the data as the training data
train_ds_mt = ds_mt["valid"].select(range(int(len(ds_mt["valid"]) * 0.8)))
train_ds_pg = ds_pg["valid"].select(range(int(len(ds_pg["valid"]) * 0.8)))
train_ds_dm = ds_dm["valid"].select(range(int(len(ds_dm["valid"]) * 0.8)))

# take the last 20% of the data as the validation data
valid_ds_mt = ds_mt["valid"].select(range(range(int(len(ds_mt["valid"]) * 0.8))[-1] + 1, len(ds_mt["valid"])))
valid_ds_pg = ds_pg["valid"].select(range(range(int(len(ds_pg["valid"]) * 0.8))[-1] + 1, len(ds_pg["valid"])))
valid_ds_dm = ds_dm["valid"].select(range(range(int(len(ds_dm["valid"]) * 0.8))[-1] + 1, len(ds_dm["valid"])))

# combine subsets into one dataset
train_ds = concatenate_datasets([train_ds_mt, train_ds_pg, train_ds_dm])
valid_ds = concatenate_datasets([valid_ds_mt, valid_ds_pg, valid_ds_dm])

# shuffling dataset
train_ds = train_ds.shuffle(seed=1337)
valid_ds = valid_ds.shuffle(seed=1337)
"""
    Next we will sort the data from easy to hard in accordance to Curriculum Learning [1]. Intuitively, hard data points are
    the ones for which humans don't have high agreement and easier samples are the ones where they do.
    First, let's check how many annotations per data sample exists.
"""
print(set(len(element) for element in ds["valid"]["labels"]))  # returns 5

"""
This only returns 5, which means that every single data point has exactly 5 annotations. Given this result, only
5 different hallucination probabilties exist, namely:
    - 0.0 where every labeled it as "No Hallucination" 
    - 0.2 where one labeled it as "Hallucination" and the rest "No Hallucination"
    - 0.4 where two labeled it as "Hallucination" and the rest "No Hallucination"
    - 0.6 where three labeled it as "Hallucination" and the rest "No Hallucination"
    - 0.8 where four labeled it as "Hallucination" and the rest "No Hallucination"
    - 1.0 where all labeled it as "Hallucination"
    
Given this result, we can group the tasks in the three difficulty levels, namely:
    - easy for all samples with Hallucination Probability of 0.0 and 1.0
    - medium for all samples with Hallucination Probability of 0.2 and 0.8
    - hard for all samples with Hallucination Probability of 0.4 and 0.6
    
Now let's group these samples into their respective difficulty.
"""

easy_train_ds = train_ds.filter(lambda x: x['p(Hallucination)'] == 0.0 or x['p(Hallucination)'] == 1.0)
easy_valid_ds = valid_ds.filter(lambda x: x['p(Hallucination)'] == 0.0 or x['p(Hallucination)'] == 1.0)

medium_train_ds = train_ds.filter(lambda x: x['p(Hallucination)'] == 0.2 or x['p(Hallucination)'] == 0.8)
medium_valid_ds = valid_ds.filter(lambda x: x['p(Hallucination)'] == 0.2 or x['p(Hallucination)'] == 0.8)

hard_train_ds = train_ds.filter(lambda x: x['p(Hallucination)'] == 0.4 or x['p(Hallucination)'] == 0.6)
hard_valid_ds = valid_ds.filter(lambda x: x['p(Hallucination)'] == 0.4 or x['p(Hallucination)'] == 0.6)

print(f"Number of easy train samples: {len(easy_train_ds)}")  # returns 145
print(f"Number of medium train samples: {len(medium_train_ds)}")  # returns 132
print(f"Number of hard train samples: {len(hard_train_ds)}")  # returns 121

print(f"Number of easy dev samples: {len(easy_valid_ds)}")  # returns 38
print(f"Number of medium dev samples: {len(medium_valid_ds)}")  # returns 39
print(f"Number of hard dev samples: {len(hard_valid_ds)}")  # returns 24

"""
For the train set, the difficulty level are somewhat balanced with 145 easy samples, 132 medium samples and 121 hard 
samples. In the dev set there is more of an unbalance but as this set is only for evaluation, it does not matter that
much.

To make use of the more fine-grained hallucination probability signal in comparison to the binary label "Hallucination" 
or "No Hallucination," we will train a model that outputs a continuous score corresponding that should correspond to the 
hallucination probability instead of a binary classifier. For that, we are using the Mean Squared Error as a loss
function.

After giving it a little bit more thought, I believe that my intuition that harder tasks correspond to the probabilities
0.4 and 0.6 could be wrong. The reason for that is such cases means that humans often disagree with whether the input is
hallucination or not. It does not mean that the sample is necessarily more complex, but it might have more to do with 
the subjective definition on what counts as hallucination or not.

To test whether whether using the different difficulty sets for training is actually useful, we will train on a model
on each difficulty set where the validation set consists of the validation data and the other train data.

As for the model we will use "OpenAssistant/reward-model-deberta-v3-base" [2] which was trained to be used as a Reward 
Model for Reinforcement Learning with Human Feedback. As my model also should output a score to predict whether the
input contains hallucination or not, this model should serve as a good baseline as this was trained on human feedback 
data.

To make most use of the model, we will flip the hallucination probabilities in our dataset, because the reward model
outputs high scores for text which it thinks are preferred by humans. As no hallucination is preferred in our case, 
a score of 1.0 should indicate preference (no hallucination) and 0.0 for disapproval (hallucination).

The input will be given in the following template: "task: {task} src: {src} tgt: {tgt} hyp: {hyp}"
"""

easy_train_ds = easy_train_ds.map(flip_function)
medium_train_ds = medium_train_ds.map(flip_function)
hard_train_ds = hard_train_ds.map(flip_function)
easy_valid_ds = easy_valid_ds.map(flip_function)
medium_valid_ds = medium_valid_ds.map(flip_function)
hard_valid_ds = hard_valid_ds.map(flip_function)
valid_ds = valid_ds.map(flip_function)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('OpenAssistant/reward-model-deberta-v3-large-v2')
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=1)
lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=128, lora_alpha=128, use_rslora=True, target_modules="all-linear")
model = get_peft_model(model, lora_config)
# Save the initial state of the model for resetting training
initial_model_state = model.state_dict()


# Tokenization function
def tokenize_function(examples):
    inputs = [f"task: {task} src: {src} tgt: {tgt} hyp: {hyp}" for task, src, tgt, hyp in
              zip(examples['task'], examples['src'], examples['tgt'], examples['hyp'])]
    return tokenizer(inputs, padding="max_length", truncation=True, max_length=512)  # Adjust max_length as needed


# Tokenize datasets
easy_train_ds = easy_train_ds.map(tokenize_function, batched=True)
medium_train_ds = medium_train_ds.map(tokenize_function, batched=True)
hard_train_ds = hard_train_ds.map(tokenize_function, batched=True)
easy_valid_ds = easy_valid_ds.map(tokenize_function, batched=True)
medium_valid_ds = medium_valid_ds.map(tokenize_function, batched=True)
hard_valid_ds = hard_valid_ds.map(tokenize_function, batched=True)
valid_ds = valid_ds.map(tokenize_function, batched=True)

easy_train_ds = easy_train_ds.filter(lambda x: len(x["input_ids"]) <= tokenizer.model_max_length)
medium_train_ds = medium_train_ds.filter(lambda x: len(x["input_ids"]) <= tokenizer.model_max_length)
hard_train_ds = hard_train_ds.filter(lambda x: len(x["input_ids"]) <= tokenizer.model_max_length)
easy_valid_ds = easy_valid_ds.filter(lambda x: len(x["input_ids"]) <= tokenizer.model_max_length)
medium_valid_ds = medium_valid_ds.filter(lambda x: len(x["input_ids"]) <= tokenizer.model_max_length)
hard_valid_ds = hard_valid_ds.filter(lambda x: len(x["input_ids"]) <= tokenizer.model_max_length)
valid_ds = valid_ds.filter(lambda x: len(x["input_ids"]) <= tokenizer.model_max_length)

# Rename p(Hallucination) to labels for compatibility with the Trainer
easy_train_ds = easy_train_ds.rename_column("labels", "old_labels")
medium_train_ds = medium_train_ds.rename_column("labels", "old_labels")
hard_train_ds = hard_train_ds.rename_column("labels", "old_labels")
easy_valid_ds = easy_valid_ds.rename_column("labels", "old_labels")
medium_valid_ds = medium_valid_ds.rename_column("labels", "old_labels")
hard_valid_ds = hard_valid_ds.rename_column("labels", "old_labels")
valid_ds = valid_ds.rename_column("labels", "old_labels")

easy_train_ds = easy_train_ds.rename_column("p(NoHallucination)", "labels")
medium_train_ds = medium_train_ds.rename_column("p(NoHallucination)", "labels")
hard_train_ds = hard_train_ds.rename_column("p(NoHallucination)", "labels")
easy_valid_ds = easy_valid_ds.rename_column("p(NoHallucination)", "labels")
medium_valid_ds = medium_valid_ds.rename_column("p(NoHallucination)", "labels")
hard_valid_ds = hard_valid_ds.rename_column("p(NoHallucination)", "labels")
valid_ds = valid_ds.rename_column("p(NoHallucination)", "labels")

# Set format for PyTorch
easy_train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
medium_train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
hard_train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
easy_valid_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
medium_valid_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
hard_valid_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
valid_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    eval_steps=1,
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    logging_first_step=True,
    save_strategy="no",
    gradient_accumulation_steps=4,
    fp16=True
)


# Custom callback to track accuracies (Generated via GPT-4o)
class AccuracyCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            epoch_accuracies[args.seed].append(metrics["eval_accuracy"])


# Load metric
metric = evaluate.load("evaluate-metric/accuracy")

# Define compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    scores.append(predictions)
    binary_labels = (labels >= 0.5).astype(int)
    binary_predictions = (predictions >= 0.5).astype(int)
    accuracy = metric.compute(predictions=binary_predictions, references=binary_labels)["accuracy"]
    return {"accuracy": accuracy}


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=interleave_datasets([easy_train_ds, medium_train_ds, hard_train_ds],
                                      stopping_strategy="all_exhausted"),
    eval_dataset=concatenate_datasets([valid_ds]),
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[AccuracyCallback()]
)
scores = []
seeds = [42, 123, 333, 666, 1337]

# only easy
if not preload_significance_values:
    epoch_accuracies = {seed: [] for seed in seeds}
    print("Training on easy samples...")
    trainer.train_dataset = easy_train_ds
    for seed in seeds:
        model.load_state_dict(initial_model_state)
        trainer.args.seed = seed
        trainer.train()
    with open(os.path.join('results', 'epoch_accuracies_easy.json'), 'w') as f:
        json.dump(epoch_accuracies, f)
else:
    with open(os.path.join('results', 'epoch_accuracies_easy.json'), 'r') as f:
        epoch_accuracies = json.load(f)
plot_accuracies(epoch_accuracies, "Training on easy train samples only")

# only medium
if not preload_significance_values:
    epoch_accuracies = {seed: [] for seed in seeds}
    print("Training on medium samples...")
    trainer.train_dataset = medium_train_ds
    for seed in seeds:
        model.load_state_dict(initial_model_state)
        trainer.args.seed = seed
        trainer.train()
    with open(os.path.join('results', 'epoch_accuracies_medium.json'), 'w') as f:
        json.dump(epoch_accuracies, f)
else:
    with open(os.path.join('results', 'epoch_accuracies_medium.json'), 'r') as f:
        epoch_accuracies = json.load(f)
plot_accuracies(epoch_accuracies, "Training on medium samples only")

# only hard
if not preload_significance_values:
    epoch_accuracies = {seed: [] for seed in seeds}
    print("Training on hard samples...")
    trainer.train_dataset = hard_train_ds
    for seed in seeds:
        model.load_state_dict(initial_model_state)
        trainer.args.seed = seed
        trainer.train()
    # Save epoch_accuracies to a JSON file
    with open(os.path.join('results', 'epoch_accuracies_hard.json'), 'w') as f:
        json.dump(epoch_accuracies, f)
else:
    with open(os.path.join('results', 'epoch_accuracies_hard.json'), 'r') as f:
        epoch_accuracies = json.load(f)
plot_accuracies(epoch_accuracies, "Training on hard samples only")


# Train on mixed dataset
if not preload_significance_values:
    epoch_accuracies = {seed: [] for seed in seeds}
    trainer.train_dataset = interleave_datasets([easy_train_ds, medium_train_ds, hard_train_ds],
                                                stopping_strategy="all_exhausted")
    print("Training on all mixed data")
    for seed in seeds:
        model.load_state_dict(initial_model_state)
        trainer.args.seed = seed
        trainer.train()
    with open(os.path.join('results', 'epoch_accuracies_mixed.json'), 'w') as f:
        json.dump(epoch_accuracies, f)
else:
    with open(os.path.join('results', 'epoch_accuracies_mixed.json'), 'r') as f:
        epoch_accuracies = json.load(f)
plot_accuracies(epoch_accuracies, "Training on all mixed style")

# Training mixed style without hard samples
if not preload_significance_values:
    epoch_accuracies = {seed: [] for seed in seeds}
    trainer.train_dataset = interleave_datasets([easy_train_ds, medium_train_ds],
                                                stopping_strategy="all_exhausted")
    for seed in tqdm.tqdm(seeds, desc="Seed Processed"):
        model.load_state_dict(initial_model_state)
        trainer.args.seed = seed
        trainer.train()
    with open(os.path.join('results', 'epoch_accuracies_mixed_wo_hard.json'), 'w') as f:
        json.dump(epoch_accuracies, f)
else:
    with open(os.path.join('results', 'epoch_accuracies_mixed_wo_hard.json'), 'r') as f:
        epoch_accuracies = json.load(f)
plot_accuracies(epoch_accuracies, "Training on all but hard samples mixed style")
mixed_data_wo_hard = {key: values[-1] for key, values in epoch_accuracies.items()}
mixed_data_wo_hard = list(mixed_data_wo_hard.values())

# curriculum style
if not preload_significance_values:
    epoch_accuracies = {seed: [] for seed in seeds}
    for seed in seeds:
        model.load_state_dict(initial_model_state)
        trainer.args.seed = seed
        trainer.train_dataset = easy_train_ds
        trainer.train()
        trainer.train_dataset = medium_train_ds
        trainer.train()
        trainer.train_dataset = hard_train_ds
        trainer.train()
    with open(os.path.join('results', 'epoch_accuracies_curriculum.json'), 'w') as f:
        json.dump(epoch_accuracies, f)
else:
    with open(os.path.join('results', 'epoch_accuracies_curriculum.json'), 'r') as f:
        epoch_accuracies = json.load(f)
plot_accuracies(epoch_accuracies, "Training Curriculum Style")

with open(os.path.join('results', 'epoch_accuracies_curriculum.json'), 'r') as f:
    epoch_accuracies_curriculum = json.load(f)

with open(os.path.join('results', 'epoch_accuracies_mixed.json'), 'r') as f:
    epoch_accuracies_mixed = json.load(f)

# Extract the last values from the first JSON file
curriculum_data = {key: values[-1] for key, values in epoch_accuracies_curriculum.items()}

# Extract the last values from the second JSON file
mixed_data = {key: values[-1] for key, values in epoch_accuracies_mixed.items()}

# Prepare the data for the t-test
curriculum_data = list(curriculum_data.values())
mixed_data = list(mixed_data.values())
# Perform paired t-test
t_statistic, p_value = stats.ttest_rel(curriculum_data, mixed_data)

print("Comparing Curriculum Style Training with all mixed data training")
print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

"""
As can be seen from the p-values and a significance level of 0.05 there is a statistically significant difference
between training the model curriculum style and on the mixed data. A negative T-statistic indicate that the average
accuracy for the mixed data is better than training curriculum style.

Interestingly to note for Curriculum Style learning is that for two seeds the performance drops a lot but recovers as
well. I am not sure what caused this phenomenon.

As can be seen from training only on the hard samples, the model struggles to perform well on the validation set. Hence,
we are going to compare the accuracy between training mixed style on all the data and without the hard samples.
"""
# Perform paired t-test
t_statistic, p_value = stats.ttest_rel(mixed_data_wo_hard, mixed_data)

print("Comparing mixed data without hard samples and mixed data with all samples")
print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

"""
The t-statistic value is negative which indicates that accuracy on the mixed data training without hard
samples is on average better than the mixed data. However, with a p-value of 0.5660 this result is not significant. 
Nevertheless, givevn the negative t-statistic value and the the performance on training only on the hard samples, we 
will discard training on the hard samples for the classifier.

So now, we will train a classifier for synthesizing the labels in the unlabeled training data. 
"""

# Training on all mixed data again
if not skip_training_classifier:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=interleave_datasets([easy_train_ds, medium_train_ds, easy_valid_ds, medium_valid_ds],
                                          stopping_strategy="all_exhausted"),
        eval_dataset=concatenate_datasets([valid_ds]),
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    print("Training on all but hard samples mixed style")
    model.load_state_dict(initial_model_state)
    trainer.train()
    model = model.merge_and_unload()
    model.save_pretrained(os.path.join("models", f"classifier"))
else:
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join("models", f"classifier")).to("cuda")
"""
Lastly, we are adding the prediction scores to the unlabeled train set and to continue go to synthesized_training.py
"""
if not skip_synthesizing_labels:
    print("Synthesizing predictions / labels ...")
    unlabeled_train_ds = load_dataset("json", data_files={"train": os.path.join("shroom_ds", "SHROOM_unlabeled-training-data-v2", "train.model-agnostic.json")})["train"]
    unlabeled_train_ds = unlabeled_train_ds.map(tokenize_function, batched=True)
    unlabeled_train_ds = unlabeled_train_ds.filter(lambda x: len(x["input_ids"]) <= tokenizer.model_max_length)
    unlabeled_train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = DataLoader(unlabeled_train_ds, batch_size=12)
    model.eval()
    label = []
    for i, batch in tqdm.tqdm(enumerate(dataloader), desc="Processing batches", total=len(dataloader)):
        batch = {k: v.to('cuda') for k, v in batch.items()}
        # Make predictions
        with torch.no_grad():
            output = model(**batch)
        values = output.logits.squeeze().tolist()
        label.extend(values)

    labeled_train_ds = unlabeled_train_ds.add_column("prediction", label)
    labeled_train_ds.save_to_disk(os.path.join("shroom_ds", f"labeled_train_ds"))

"""
References:
h
[2] https://huggingface.co/OpenAssistant/reward-model-deberta-v3-base
"""
