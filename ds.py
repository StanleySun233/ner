from collections import defaultdict
from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value
from huggingface_hub import HfApi

# 初始化 API
api = HfApi()
names = ['resume','msra','weibo']

def load_data(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        label = []
        for line in f:
            line = line.strip()
            if not line:
                if sentence:  # 每次空行表示一个句子结束
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []
                    label = []
            else:
                char, ner_label = line.split()
                sentence.append(char)
                label.append(ner_label)
        if sentence:  # 将最后一个句子也加入
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels


def generate_label_list(sentences, labels):
    label_set = set()
    for ner_tags in labels:
        for tag in ner_tags:
            label_set.add(tag)
    entity_types = defaultdict(set)
    for label in label_set:
        if label == "O":
            continue
        prefix, entity = label.split("-")
        entity_types[entity].add(prefix)
    _label_list = ["O"]
    for entity in sorted(entity_types):
        if "B" in entity_types[entity]:
            _label_list.append(f"B-{entity}")
        if "I" in entity_types[entity]:
            _label_list.append(f"I-{entity}")
    return _label_list

for name in names:
    train_sentences, train_labels = load_data(f"./data/{name}/train.txt")
    test_sentences, test_labels = load_data(f"./data/{name}/test.txt")
    validation_sentences, validation_labels = load_data(f"./data/{name}/dev.txt")

    label_list = generate_label_list(train_sentences,train_labels)

    # 定义 ClassLabel 来自动将标签转为 ID
    label_class = ClassLabel(names=label_list)


    # 将标签从字符串转换为 ID
    def convert_labels_to_ids(labels):
        return [[label_class.str2int(label) for label in ner_tag] for ner_tag in labels]


    # 将训练、测试和验证标签都转换为 ID
    train_labels = convert_labels_to_ids(train_labels)
    test_labels = convert_labels_to_ids(test_labels)
    validation_labels = convert_labels_to_ids(validation_labels)

    # 定义特征
    features = Features({
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=label_list))
    })

    # 将数据转化为 Hugging Face 的 Dataset 格式
    train_data = Dataset.from_dict({"tokens": train_sentences, "ner_tags": train_labels},features=features)

    test_data = Dataset.from_dict({"tokens": test_sentences, "ner_tags": test_labels},features=features)

    validation_data = Dataset.from_dict({"tokens": validation_sentences, "ner_tags": validation_labels},features=features)


    # 合并为 DatasetDict 格式
    ner_dataset = DatasetDict({"train": train_data, "test": test_data, "validation": validation_data})

    # 定义数据集名称和描述
    dataset_name = f"{name}-ner"  # 数据集名称，可以根据实际需求修改

    # 推送数据集
    ner_dataset.push_to_hub(dataset_name, token="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")
