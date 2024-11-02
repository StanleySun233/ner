from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import Trainer,TrainingArguments,TrainerCallback
import numpy as np
import evaluate
from torch.utils.tensorboard import SummaryWriter

dataset = load_dataset("conll2003",trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased",
                                                        num_labels=len(dataset['train'].features['ner_tags'].feature.names))

label_list = dataset["train"].features["ner_tags"].feature.names
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer)

metric = evaluate.load("seqeval")
writer = SummaryWriter(log_dir="./logs")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    label_list = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER",'B-ORG','I-ORG','B-LOC','I-LOC']  # Replace with your actual label list

    results = metric.compute(predictions=predictions, references=labels)

    # Log overall metrics
    writer.add_scalar("Overall/Precision", results["overall_precision"], global_step=trainer.state.global_step)
    writer.add_scalar("Overall/Recall", results["overall_recall"], global_step=trainer.state.global_step)
    writer.add_scalar("Overall/F1", results["overall_f1"], global_step=trainer.state.global_step)

    # Log per-class metrics
    for label, metrics in results.items():
        if isinstance(metrics, dict):  # Filter out "overall" metrics
            writer.add_scalar(f"Class_{label}/Precision", metrics["precision"], global_step=trainer.state.global_step)
            writer.add_scalar(f"Class_{label}/Recall", metrics["recall"], global_step=trainer.state.global_step)
            writer.add_scalar(f"Class_{label}/F1", metrics["f1"], global_step=trainer.state.global_step)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

class TensorBoardCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # 将每次日志的损失值记录到 TensorBoard
        for k, v in logs.items():
            if "loss" in k:
                writer.add_scalar(f"Loss/{k}", v, global_step=state.global_step)


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",                # 指定日志目录
    logging_strategy="steps",            # 设置日志记录策略（这里按步记录）
    logging_steps=10,                    # 每 10 步记录一次日志
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=True,
    hub_model_id="PassbyGrocer/bert-ner-conll2003"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[TensorBoardCallback()]
)

trainer.train()

trainer.push_to_hub()