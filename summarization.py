from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset
from datasets.utils.logging import set_verbosity_info
import evaluate
import torch
import shutil
import os

torch.cuda.empty_cache()

set_verbosity_info()

cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")

if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

try:
    dataset = load_dataset("cnn_dailymail", "3.0.0")
except Exception as e:
    print(f"Error loading dataset: {e}")
    dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir="./cache")

def preprocess_data(examples):
    model_inputs = tokenizer(examples['article'], max_length=128, truncation=True, padding='max_length')
    labels = tokenizer(examples['highlights'], max_length=100, truncation=True, padding='max_length').input_ids
    model_inputs['labels'] = labels
    return model_inputs

summarization_models = [
    "facebook/bart-base", 
    "google/pegasus-xsum", 
    "t5-small",
    "sshleifer/distilbart-cnn-12-6",
    "philschmid/bart-large-cnn-samsum"
]

for model_name in summarization_models:
    print(f"Fine-tuning {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model.gradient_checkpointing_enable()

    tokenized_dataset = dataset.map(preprocess_data, batched=True)

    training_args = TrainingArguments(
        output_dir=f"./results_{model_name}",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=3e-5,
        warmup_steps=500,
        logging_dir=f"./logs_{model_name}",
        fp16=True,
        lr_scheduler_type='cosine',
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'].shuffle(seed=42).select(range(5000)),
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer,
    )

    trainer.train()

    torch.cuda.empty_cache()

    model.to('cpu')

    rouge = evaluate.load("rouge")

    validation_subset = tokenized_dataset['validation'].select(range(500))

    def generate_summary(batch):
        inputs = tokenizer(batch['article'], max_length=128, truncation=True, padding='max_length', return_tensors='pt').to('cpu')
        with torch.no_grad():
            summaries = model.generate(inputs['input_ids'], max_length=100, num_beams=5, early_stopping=True)
        return {'generated_summary': [tokenizer.decode(g, skip_special_tokens=True) for g in summaries]}

    results = validation_subset.map(generate_summary, batched=True, batch_size=16)

    rouge_scores = rouge.compute(predictions=results['generated_summary'], references=results['highlights'])

    print(f"ROUGE scores for {model_name}: {rouge_scores}")
