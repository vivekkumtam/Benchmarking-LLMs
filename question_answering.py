# Install required libraries
!pip install datasets transformers evaluate

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import torch

torch.cuda.empty_cache()

dataset = load_dataset("squad_v2")

def preprocess_data(examples, tokenizer):
    inputs = tokenizer(
        examples['question'],
        examples['context'],
        padding=True,
        truncation=True
    )

    start_positions = []
    end_positions = []

    for i in range(len(examples["answers"])):
        if len(examples["answers"][i]["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_position = examples["answers"][i]["answer_start"][0]
            end_position = start_position + len(examples["answers"][i]["text"][0])

            tokenized_context = tokenizer(examples["context"][i], padding=True, truncation=True)
            start_token = tokenized_context.char_to_token(start_position)
            end_token = tokenized_context.char_to_token(end_position - 1)

            start_positions.append(start_token if start_token is not None else 0)
            end_positions.append(end_token if end_token is not None else 0)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs

qa_models = [
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "albert-base-v2",
    "distilbert-base-cased-distilled-squad",
    "distilroberta-base",
    "xlnet-base-cased"
]

squad_metric = evaluate.load("squad_v2")

for model_name in qa_models:
    print(f"\nFine-tuning {model_name}...\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    tokenized_dataset = dataset.map(lambda x: preprocess_data(x, tokenizer), batched=True)
    train_dataset = tokenized_dataset['train'].select(range(20))
    eval_dataset = tokenized_dataset['validation'].select(range(10))

    training_args = TrainingArguments(
        output_dir=f"./results_{model_name}",
        evaluation_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        logging_dir=f"./logs_{model_name}",
        fp16=True,
        report_to="none",
        save_safetensors=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    torch.cuda.empty_cache()

    with torch.no_grad():
        raw_predictions = trainer.predict(eval_dataset)

    decoded_predictions = []
    for i in range(len(eval_dataset)):
        start_idx = raw_predictions.predictions[0][i].argmax()
        end_idx = raw_predictions.predictions[1][i].argmax()
        prediction_text = tokenizer.decode(eval_dataset['input_ids'][i][start_idx:end_idx+1], skip_special_tokens=True)
        decoded_predictions.append({"id": str(i), "prediction_text": prediction_text, "no_answer_probability": 0.0})

    references = [{"id": str(i), "answers": example["answers"]} for i, example in enumerate(dataset['validation'].select(range(10)))]

    results = squad_metric.compute(predictions=decoded_predictions, references=references)
    print(f"Results for {model_name}: {results}")
