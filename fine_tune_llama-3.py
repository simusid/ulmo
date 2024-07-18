import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Replace with the correct model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the pad_token to the eos_token if it's not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id


# Load and preprocess the data
with open('/Users/gary/Downloads/data.json', 'r') as f:
    data = json.load(f)

def preprocess_function(examples):
    inputs = [f"### Input: {input_text}\n### Output: " for input_text in examples['input']]
    targets = examples['output']
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")["input_ids"]
    
    model_inputs["labels"] = labels
    return model_inputs

# Create a Dataset object
dataset = Dataset.from_dict({"input": [item['input'] for item in data], "output": [item['output'] for item in data]})
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Start training
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_llama")