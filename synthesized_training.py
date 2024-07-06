import os
from datasets import load_from_disk, load_dataset
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding
import evaluate
from utils import percentile_score

########### Run Parameters ####################################
# version 0: 'OpenAssistant/reward-model-deberta-v3-base'
# version 1: 'models/synthesized_model_v1'
# version 2: 'models/synthesized_model_v2'
checkpoint = 'OpenAssistant/reward-model-deberta-v3-base'
skip_training = False  # if skip_training is True the current checkpoint will be used on the test set
version = 1  # this value must be +1 the checkpoint version! (only considered when skip_training=False)
evaluate_on_test_set = True
################################################################

########## Hyperparameters #####################################
n = 5000  # get the top and bottom n values of the prediction list
################################################################


# Load the accuracy metric
metric = evaluate.load("evaluate-metric/accuracy")

# Load the dataset
ds = load_from_disk(os.path.join("shroom_ds", f"labeled_train_ds_v{version}"))

"""
Given our new train set with the prediction, lets first plot the distribution of the scores.
"""
# Plot the histogram of predictions
plt.hist(ds["prediction"], bins=10, edgecolor='black')  # Adjust the number of bins as needed
plt.title('Score Distribution')
plt.xlabel('Score Range')
plt.ylabel('Frequency')
plt.show()

"""
As can be seen from the plot, there are way less scores below 0.5 than above 0.5 indicating way less hallucination
predictions than not hallucination. This suggests, that the model is really confident predicting no hallucinations,
whereas it is unsure when predicting hallucinations.
"""
# convert percentile thresholds into score thresholds
no_hallucincation_threshold, hallucination_threshold = percentile_score(ds["prediction"], n)
print(f"The score threshold for no hallucination is at {no_hallucincation_threshold}")
print(f"The score threshold for hallucination is at {hallucination_threshold}")
"""
Another thing we are going to do for the classifier is computing the accuracy on different cutoffs and then selecting
the best cutoff for the test set. The cutoff is the point where all scores above the cutoff are regarded as 
Not Hallucination and all points below as Hallucination. The Rationale is that our model does not necessarily output
scores between 0 and 1 and therefore the optimal cutoff is not necessarily on 0.5.

Additionally, to not lose the more fine-grained signal, we are not going to label the train set binarily but we are 
going to use the prediction scores instead. This makes sure that the model can still learn which samples are of better
quality, i.e., the higher the distance from the cutoff point the more confident the prediction is.
"""
cutoffs = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]


# Labeling function
def labeling(example):
    if example["prediction"] >= no_hallucincation_threshold:
        example["labels"] = 1.0
    elif example["prediction"] <= hallucination_threshold:
        example["labels"] = 0.0
    else:
        example["labels"] = -1.0
    return example


# Convert labels to integers
def label2int(example):
    if example["label"] == "Not Hallucination":
        example["label"] = 1.0
    else:
        example["label"] = 0.0
    return example


# Compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    output = {}
    best_cutoff = None
    best_accuracy = 0

    for cutoff in cutoffs:
        binary_predictions = (predictions >= cutoff).astype(int)
        accuracy = metric.compute(predictions=binary_predictions, references=labels)["accuracy"]
        output[f"accuracy_{cutoff}"] = accuracy

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_cutoff = cutoff

    output["best_cutoff"] = best_cutoff
    return output


def compute_metrics_for_test_set(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    output = {}
    binary_predictions = (predictions >= cutoff).astype(int)
    accuracy = metric.compute(predictions=binary_predictions, references=labels)["accuracy"]
    output[f"accuracy"] = accuracy
    return output


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('OpenAssistant/reward-model-deberta-v3-base')
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Tokenization function
def tokenize_function(examples):
    inputs = [f"task: {task} src: {src} tgt: {tgt} hyp: {hyp}" for task, src, tgt, hyp in
              zip(examples['task'], examples['src'], examples['tgt'], examples['hyp'])]
    return tokenizer(inputs, padding="max_length", truncation=True, max_length=512)


# Apply labeling and filtering
# ds = ds.map(labeling)
ds = ds.filter(lambda x: x["prediction"] >= no_hallucincation_threshold or x["prediction"] <= hallucination_threshold)
ds = ds.rename_column("prediction", "labels")
ds = ds.shuffle(seed=1337)
print(f"Number of Not Hallucination: {len([x for x in ds['labels'] if x >= no_hallucincation_threshold])}")
print(f"Number of Hallucination: {len([x for x in ds['labels'] if x <= hallucination_threshold])}")

# Load and preprocess test dataset
ds_valid = \
load_dataset("json", data_files={"valid": os.path.join("shroom_ds", "SHROOM_dev-v2", "val.model-agnostic.json")})[
    "valid"]
ds_valid = ds_valid.map(tokenize_function, batched=True)
ds_valid = ds_valid.map(label2int)
ds_valid = ds_valid.remove_columns(["labels"])
ds_valid = ds_valid.rename_column("label", "labels")
ds_valid.set_format(type='torch', columns=['input_ids', 'attention_mask', "labels"])

# Load and preprocess test dataset
ds_test = \
load_dataset("json", data_files={"test": os.path.join("shroom_ds", "SHROOM_test-labeled", "test.model-agnostic.json")})[
    "test"]
ds_test = ds_test.map(tokenize_function, batched=True)
ds_test = ds_test.map(label2int)
ds_test = ds_test.remove_columns(["labels"])
ds_test = ds_test.rename_column("label", "labels")
ds.set_format(type='torch', columns=['input_ids', 'attention_mask', "labels"])
ds_test.set_format(type='torch', columns=['input_ids', 'attention_mask', "labels"])

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    eval_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_steps=100,
    logging_first_step=True,
    save_strategy="no",
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    eval_dataset=ds_valid,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# Train the model
if not skip_training:
    trainer.train()
    trainer.save_model(os.path.join("models", f"synthesized_model_v{version}"))

# Leverage quality samples
train_ds = trainer.eval_dataset
train_ds = train_ds.filter()

output = trainer.evaluate()
print("============Results on the validation set============")
print(output)
cutoff = output["eval_best_cutoff"]
print(f"Selecting best cutoff at {cutoff} to use on test set")

if evaluate_on_test_set:
    trainer.compute_metrics = compute_metrics
    trainer.eval_dataset = ds_test
    output = trainer.evaluate()
    print("============Results on the test set============")
    print(output)
