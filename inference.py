import os
import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader


# Hyperparameter
batch_size = 16
model_name = "synthesized_model"

tokenizer = AutoTokenizer.from_pretrained('OpenAssistant/reward-model-deberta-v3-large-v2')
model = AutoModelForSequenceClassification.from_pretrained(os.path.join('models', model_name)).to("cuda")
dev_set = load_dataset("json",
                       data_files={"test": os.path.join("shroom_ds", "SHROOM_test-labeled", "test.model-agnostic.json")},
                       split="test")

def tokenize_function(examples):
    inputs = [f"task: {task} src: {src} tgt: {tgt} hyp: {hyp}" for task, src, tgt, hyp in
              zip(examples['task'], examples['src'], examples['tgt'], examples['hyp'])]
    return tokenizer(inputs, padding="max_length", truncation=True, max_length=512)

dev_set = dev_set.map(tokenize_function, batched=True)
dev_set.set_format(type='torch', columns=['input_ids', 'attention_mask'])
dataloader = DataLoader(dev_set, batch_size=batch_size)

model.eval()

# Process batches and make predictions
predictions = []
for batch in tqdm.tqdm(dataloader, desc="Processing batches"):
    batch = {k: v.to('cuda') for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits.squeeze().tolist()
    predictions.extend(logits if isinstance(logits, list) else [logits])

# Add predictions to the dataset
dev_set = dev_set.add_column("prediction", predictions)
df = dev_set.to_pandas()

# Plotting the histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="prediction", hue="label", kde=False, bins=50, palette="viridis", multiple="stack")
plt.title(f'Score Distribution by Label ({model_name})')
plt.xlabel('Score')
plt.ylabel('Frequency')

plt.savefig(os.path.join("plots", f"score_distribution_{model_name}.png"))
plt.show()

