import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
import torch
from transformers import BitsAndBytesConfig

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Replace with the correct model name

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Set the pad_token to the eos_token if it's not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Prepare the model for k-bit training
model = prepare_model_for_kbit_training(model)

# Define LoRA Config
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

# Get the PEFT model
model = get_peft_model(model, peft_config)

def preprocess_function(examples):
    inputs = [f"### Input: {input_text}\n### Output: " for input_text in examples['input']]
    targets = examples['output']
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")["input_ids"]
    
    model_inputs["labels"] = labels
    return model_inputs

# Load and preprocess the data
with open('data.json', 'r') as f:
    data = json.load(f)

# Create a Dataset object
dataset = Dataset.from_dict({"input": [item['input'] for item in data], "output": [item['output'] for item in data]})
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    save_steps=10,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    optim="paged_adamw_8bit"
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
trainer.model.save_pretrained("./fine_tuned_llama_peft")