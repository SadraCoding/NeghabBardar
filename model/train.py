from transformers import AutoModelForImageClassification, AutoFeatureExtractor, Trainer, TrainingArguments, DefaultDataCollator
from datasets import Dataset, Image, concatenate_datasets, load_from_disk
import torch
import os
import shutil
import numpy as np
from sklearn.metrics import accuracy_score

model_dir = "Organika/sdxl-detector"
output_dir = "./sdxl-detector-finetuned"
os.makedirs(output_dir, exist_ok=True)

def list_image_paths_and_labels(root_dir):
    classes = sorted(os.listdir(root_dir))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    images, labels = [], []
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        for filename in os.listdir(cls_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                images.append(os.path.join(cls_dir, filename))
                labels.append(class_to_idx[cls])
    return images, labels

train_image_dir = "./dataset/train"
val_image_dir = "./dataset/val"
train_images, train_labels = list_image_paths_and_labels(train_image_dir)
val_images, val_labels = list_image_paths_and_labels(val_image_dir)

train_dataset = Dataset.from_dict({"image": train_images, "label": train_labels}).cast_column("image", Image())
val_dataset = Dataset.from_dict({"image": val_images, "label": val_labels}).cast_column("image", Image())

feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)

def preprocess_function(examples):
    inputs = feature_extractor(examples["image"], return_tensors="pt")
    inputs["labels"] = examples["label"]
    return {k: v.numpy() if torch.is_tensor(v) else v for k, v in inputs.items()}

def process_and_save_chunk(dataset, chunk_index, chunk_size, base_cache_dir):
    start = chunk_index * chunk_size
    end = min(start + chunk_size, len(dataset))
    print(f"Processing chunk {chunk_index + 1}, samples {start} to {end}")
    chunk = dataset.select(range(start, end))
    chunk = chunk.map(preprocess_function, batched=True, batch_size=1, num_proc=1, load_from_cache_file=False)
    
    chunk_cache_dir = os.path.join(base_cache_dir, f"chunk_{chunk_index}")
    if os.path.exists(chunk_cache_dir):
        shutil.rmtree(chunk_cache_dir)
    chunk.save_to_disk(chunk_cache_dir)
    chunk.cleanup_cache_files()
    return chunk_cache_dir

def load_and_concat_chunks(base_cache_dir, num_chunks):
    chunk_datasets = []
    for i in range(num_chunks):
        chunk_cache_dir = os.path.join(base_cache_dir, f"chunk_{i}")
        print(f"Loading chunk dataset from {chunk_cache_dir}")
        ds_chunk = load_from_disk(chunk_cache_dir)
        chunk_datasets.append(ds_chunk)
    print("Concatenating all chunk datasets...")
    full_dataset = concatenate_datasets(chunk_datasets)
    return full_dataset

chunk_cache_base_dir = "./temp_chunks"
os.makedirs(chunk_cache_base_dir, exist_ok=True)

chunk_size = 20000
num_train_chunks = (len(train_dataset) + chunk_size -1) // chunk_size
num_val_chunks = (len(val_dataset) + chunk_size -1) // chunk_size

for i in range(num_train_chunks):
    process_and_save_chunk(train_dataset, i, chunk_size, chunk_cache_base_dir)
for i in range(num_val_chunks):
    process_and_save_chunk(val_dataset, i, chunk_size, chunk_cache_base_dir)

train_dataset_processed = load_and_concat_chunks(chunk_cache_base_dir, num_train_chunks)
val_dataset_processed = load_and_concat_chunks(chunk_cache_base_dir, num_val_chunks)
shutil.rmtree(chunk_cache_base_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForImageClassification.from_pretrained(model_dir).to(device)
data_collator = DefaultDataCollator()

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    report_to=[],
    save_total_limit=2,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    print(f"Accuracy: {acc:.4f}")
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_processed,
    eval_dataset=val_dataset_processed,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor,
)

trainer.train()
trainer.save_model(output_dir)
print(f"Model saved at {output_dir}")
