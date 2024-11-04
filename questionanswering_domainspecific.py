# Install required libraries
!pip install datasets transformers evaluate

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
import evaluate
import torch

torch.cuda.empty_cache()

class QA_Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        start_positions = inputs.pop("start_positions")
        end_positions = inputs.pop("end_positions")
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        start_loss = torch.nn.functional.cross_entropy(outputs.start_logits, start_positions)
        end_loss = torch.nn.functional.cross_entropy(outputs.end_logits, end_positions)
        loss = (start_loss + end_loss) / 2
        return (loss, outputs) if return_outputs else loss

model_dataset_pairs = {
    "bert-large-uncased-whole-word-masking-finetuned-squad": {
        "dataset": "squad_v2",
        "question_column": "question",
        "context_column": "context",
        "answer_column": "answers"
    },
    "albert-base-v2": {
        "dataset": "financial_phrasebank/sentences_allagree",
        "question_column": None,
        "context_column": "sentence",
        "answer_column": None
    },
    "distilbert-base-cased-distilled-squad": {
        "dataset": "medmcqa",
        "question_column": "question",
        "context_column": "context",
        "answer_column": "answers"
    },
    "distilroberta-base": {
        "dataset": "scifact/claims",
        "question_column": "claim",
        "context_column": "evidence",
        "answer_column": "evidence"
    },
    "xlnet-base-cased": {
        "dataset": "lex_glue/ecthr_a",
        "question_column": "question",
        "context_column": "context",
        "answer_column": "answers"
    }
}

squad_metric = evaluate.load("squad_v2")

def preprocess_data(examples, tokenizer, question_column, context_column, answer_column):
    if question_column is None or question_column not in examples:
        examples["question"] = ["What information is provided here?"] * len(examples.get(context_column, ["No context"] * len(examples)))
        question_column = "question"
    if context_column is None or context_column not in examples:
        examples["context"] = ["This is placeholder context."] * len(examples[question_column])
        context_column = "context"
    if answer_column is None or answer_column not in examples:
        examples["answers"] = [{"answer_start": [0], "text": ["Placeholder answer."]}] * len(examples[context_column])
        answer_column = "answers"

    inputs = tokenizer(
        examples[question_column] if isinstance(examples[question_column], list) else [examples[question_column]],
        examples[context_column] if isinstance(examples[context_column], list) else [examples[context_column]],
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True
    )

    offset_mappings = inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mappings):
        if len(examples[answer_column][i]["text"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        answer = examples[answer_column][i]["text"][0]
        start_char = examples[answer_column][i]["answer_start"][0]
        end_char = start_char + len(answer)

        start_token = None
        end_token = None
        for j, (offset_start, offset_end) in enumerate(offsets):
            if offset_start <= start_char < offset_end:
                start_token = j
            if offset_start < end_char <= offset_end:
                end_token = j
                break

        start_positions.append(start_token if start_token is not None else 0)
        end_positions.append(end_token if end_token is not None else 0)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs

for model_name, config in model_dataset_pairs.items():
    print(f"\nFine-tuning {model_name} on {config['dataset']} dataset...\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    dataset_name, *dataset_config = config["dataset"].split('/')
    dataset = load_dataset(dataset_name, dataset_config[0] if dataset_config else None)

    if "validation" not in dataset:
        dataset = DatasetDict({
            "train": dataset["train"].train_test_split(test_size=0.1)["train"],
            "validation": dataset["train"].train_test_split(test_size=0.1)["test"]
        })

    question_column = config["question_column"]
    context_column = config["context_column"]
    answer_column = config["answer_column"]

    tokenized_dataset = dataset.map(
        lambda x: preprocess_data(x, tokenizer, question_column, context_column, answer_column),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    train_sample_size = min(20, len(tokenized_dataset["train"]))
    eval_sample_size = min(10, len(tokenized_dataset["validation"]))
    train_dataset = tokenized_dataset["train"].select(range(train_sample_size))
    eval_dataset = tokenized_dataset["validation"].select(range(eval_sample_size))

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

    trainer = QA_Trainer(
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
        prediction_text = tokenizer.decode(eval_dataset["input_ids"][i][start_idx:end_idx+1], skip_special_tokens=True)
        decoded_predictions.append({"id": str(i), "prediction_text": prediction_text, "no_answer_probability": 0.0})

    references = [
        {
            "id": str(i),
            "answers": example.get(answer_column, {"answer_start": [0], "text": ["Placeholder answer."]})
        } for i, example in enumerate(dataset["validation"].select(range(eval_sample_size)))
    ]

    results = squad_metric.compute(predictions=decoded_predictions, references=references)
    print(f"Results for {model_name} on {config['dataset']} dataset: {results}")
