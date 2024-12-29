!pip install datasets transformers evaluate seqeval

from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import load_dataset
import evaluate
import torch

def preprocess_data(examples, tokenizer, max_length=64):
    tokens = tokenizer(examples['tokens'], truncation=True, padding='max_length', max_length=max_length, is_split_into_words=True)
    labels = examples['ner_tags'] if 'ner_tags' in examples else examples['label']
    tokens['labels'] = labels
    return tokens

def train_and_evaluate_model(model_name, dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=9)

    small_train_dataset = dataset['train'].shuffle(seed=42).select(range(500))  # Use a subset of 500 samples
    small_eval_dataset = dataset['validation'].shuffle(seed=42).select(range(100))  # Use a subset of 100 samples
    tokenized_train = small_train_dataset.map(lambda x: preprocess_data(x, tokenizer), batched=True)
    tokenized_eval = small_eval_dataset.map(lambda x: preprocess_data(x, tokenizer), batched=True)
    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_eval.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=4,  
        num_train_epochs=1,  
        logging_dir="./logs",
        report_to="none",
        no_cuda=True,  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    predictions, labels, _ = trainer.predict(tokenized_eval)
    predictions = predictions.argmax(axis=-1)

    pred_labels = [[model.config.id2label[p] for p, l in zip(preds, label) if l != -100] for preds, label in zip(predictions, labels)]
    true_labels = [[model.config.id2label[l] for l in label if l != -100] for label in labels]

    seqeval_metric = evaluate.load("seqeval")
    results = seqeval_metric.compute(predictions=pred_labels, references=true_labels)
    print(f"Results for {model_name}: {results}")

dataset = load_dataset("conll2003")

model_names = ["bert-base-cased", "distilbert-base-cased", "dmis-lab/biobert-v1.1"]

for model_name in model_names:
    train_and_evaluate_model(model_name, dataset)
