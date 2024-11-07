import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments, AutoConfig


def check_torch_gpu():
    # 获取 CUDA 版本
    cuda_version = torch.version.cuda

    # 获取 cuDNN 版本
    cudnn_version = torch.backends.cudnn.version()

    # 获取 PyTorch 版本
    torch_version = torch.__version__

    # 检查是否可以在 GPU 上运行 PyTorch
    gpu_available = torch.cuda.is_available()

    print(f"CUDA Version: {cuda_version}")
    print(f"cuDNN Version: {cudnn_version}")
    print(f"PyTorch Version: {torch_version}")
    print("Is GPU available for PyTorch:", "Yes" if gpu_available else "No")
    return {
        "CUDA Version": cuda_version,
        "cuDNN Version": cudnn_version,
        "PyTorch Version": torch_version,
        "GPU Available": gpu_available
    }


def load_datasets_by_hf(dataset_name, tokenizer_name):
    dataset = load_dataset(dataset_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding='max_length',
                                     is_split_into_words=True, max_length=64)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    return tokenized_datasets, data_collator, tokenizer


def load_model_by_hf(pretrained_name, dataset, cls) -> AutoModelForTokenClassification:
    print(dataset["train"].features["ner_tags"])

    num_labels = len(dataset['train'].features['ner_tags'].feature.names)

    id2label = {i: dataset['train'].features['ner_tags'].feature.names[i] for i in range(num_labels)}
    label2id = {v: k for k, v in id2label.items()}

    # 加载配置并设置标签映射
    config = AutoConfig.from_pretrained(pretrained_name, num_labels=num_labels, id2label=id2label, label2id=label2id)

    # 初始化自定义模型
    model = cls.from_pretrained(pretrained_name, config=config)
    return model


def compute_metrics(p, _label, _metrics):
    # 获取 logits，并将它们转换为预测的标签索引
    predictions = np.argmax(p.predictions, axis=2)
    references = p.label_ids

    # 转换为字符串标签，并过滤掉 -100
    true_predictions = [
        [_label[pred] for (pred, label) in zip(prediction, reference) if label != -100]
        for prediction, reference in zip(predictions, references)
    ]

    true_labels = [
        [_label[label] for (pred, label) in zip(prediction, reference) if label != -100]
        for prediction, reference in zip(predictions, references)
    ]

    # 计算 seqeval 指标并过滤无效标签
    results = _metrics.compute(predictions=true_predictions, references=true_labels)

    # 检查并设置缺失标签的 F1 和 Recall 为 0
    for key, value in results.items():
        if value is None:
            results[key] = 1

    return {
        "precision": results.get("overall_precision", 1),
        "recall": results.get("overall_recall", 1),
        "f1": results.get("overall_f1", 1),
        "accuracy": results.get("overall_accuracy", 1),
    }


def get_train_args(user_id, model_name, dataset_name, epoches=100, batch=32):
    return TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_dir="./logs",  # 指定日志目录
        logging_strategy="steps",  # 设置日志记录策略（这里按步记录）
        logging_steps=10,  # 每 10 步记录一次日志
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        num_train_epochs=epoches,
        fp16=False,
        push_to_hub=True,
        hub_model_id=f"{user_id}/{model_name}-ner-{dataset_name}",
    )


def get_trainer(model, training_args, tokenized_datasets, tokenizer, data_collator, cm):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=cm,
        optimizers=(optimizer, None),
    )


def train(dataset_name, pretrained_name, user_id, model_name, model_cls,epoches=100,batch=32):
    tokenized_datasets, data_collator, tokenizer = load_datasets_by_hf(dataset_name, pretrained_name)
    model = load_model_by_hf(pretrained_name, tokenized_datasets, model_cls)
    metric = evaluate.load("seqeval")
    label_list = tokenized_datasets["train"].features["ner_tags"].feature.names  # 获取标签名称列表
    s = [i for i in range(len(label_list))]

    def compute_metrics_helper(pred):
        return compute_metrics(pred, label_list, metric)

    train_args = get_train_args(user_id, model_name, dataset_name.split("/")[1],epoches,batch)
    trainer = get_trainer(model, train_args, tokenized_datasets, tokenizer, data_collator, compute_metrics_helper)

    trainer.train()
    test_results = trainer.predict(tokenized_datasets["test"])
    print("Result in Test:", test_results.metrics)

    trainer.push_to_hub()


if __name__ == "__main__":
    check_torch_gpu()
