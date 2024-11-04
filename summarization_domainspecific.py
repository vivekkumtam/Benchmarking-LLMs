# Install necessary libraries
!pip install datasets transformers evaluate torch rouge_score py7zr

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import torch
import os
import shutil

torch.cuda.empty_cache()

domain_specific_configurations = {
    "scientific_distilbart": {
        "dataset": ("scientific_papers", "pubmed"),
        "model": "sshleifer/distilbart-cnn-12-6",
        "input_column": "article",
        "summary_column": "abstract"
    },
    "conversational_bart_large": {
        "dataset": ("samsum", None),
        "model": "philschmid/bart-large-cnn-samsum",
        "input_column": "dialogue",
        "summary_column": "summary"
    },
    "news_bart": {
        "dataset": ("cnn_dailymail", "3.0.0"),
        "model": "facebook/bart-base",
        "input_column": "article",
        "summary_column": "highlights"
    },
    "news_pegasus": {
        "dataset": ("xsum", None),
        "model": "google/pegasus-xsum",
        "input_column": "document",
        "summary_column": "summary"
    },
    "scientific_t5": {
        "dataset": ("scientific_papers", "pubmed"),
        "model": "t5-small",
        "input_column": "article",
        "summary_column": "abstract"
    }
}

def clear_cache():
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    torch.cuda.empty_cache()

def preprocess_data(examples, tokenizer, input_column, summary_column, max_input_length=512, max_target_length=128):
    inputs = [str(text) if text else "" for text in examples[input_column]]
    summaries = [str(text) if text else "" for text in examples[summary_column]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding='max_length')
    labels = tokenizer(summaries, max_length=max_target_length, truncation=True, padding='max_length').input_ids
    model_inputs['labels'] = labels
    return model_inputs

rouge = evaluate.load("rouge")

for domain_name, config in domain_specific_configurations.items():
    print(f"Processing domain: {domain_name}")

    dataset_name, subset = config["dataset"]
    model_name = config["model"]
    input_column = config["input_column"]
    summary_column = config["summary_column"]

    try:
        clear_cache()
        dataset = load_dataset(dataset_name, subset, trust_remote_code=True) if subset else load_dataset(dataset_name, trust_remote_code=True)
        available_columns = dataset['train'].column_names if 'train' in dataset else dataset.column_names
        if input_column not in available_columns or summary_column not in available_columns:
            print(f"Error: Specified columns not found in dataset {dataset_name}. Available columns: {available_columns}")
            continue
    except Exception as e:
        print(f"Error loading dataset for {domain_name}: {e}")
        continue

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.gradient_checkpointing_enable()

    tokenized_dataset = dataset.map(lambda x: preprocess_data(x, tokenizer, input_column, summary_column), batched=True)

    training_args = TrainingArguments(
        output_dir=f"./results_{domain_name}",
        evaluation_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        learning_rate=3e-5,
        warmup_steps=500,
        logging_dir=f"./logs_{domain_name}",
        fp16=True,
        lr_scheduler_type='cosine',
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'].shuffle(seed=42).select(range(2000)),
        eval_dataset=tokenized_dataset['validation'].select(range(500)),
        tokenizer=tokenizer,
    )

    trainer.train()

    model.to('cpu')

    def generate_summary(batch):
        inputs = tokenizer(batch[input_column], max_length=512, truncation=True, padding='max_length', return_tensors='pt').to('cpu')
        with torch.no_grad():
            summaries = model.generate(inputs['input_ids'], max_length=128, num_beams=5, early_stopping=True)
        return {'generated_summary': [tokenizer.decode(g, skip_special_tokens=True) for g in summaries]}

    validation_subset = tokenized_dataset['validation'].select(range(500))
    results = validation_subset.map(generate_summary, batched=True, batch_size=8)
    rouge_scores = rouge.compute(predictions=results['generated_summary'], references=results[summary_column])

    print(f"ROUGE scores for {domain_name} with {model_name}: {rouge_scores}")
