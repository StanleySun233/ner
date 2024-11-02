from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained("PassbyGrocer/bert-ner-conll2003")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "My name is Wolfgang and I live in Berlin."

ner_results = nlp(example)
print(ner_results)
formatted_output = ['O'] * len(example)
for entity in ner_results:
    word = entity['word']
    label = entity['entity']
    s = entity['start']
    e = entity['end']
    formatted_output[s] = label
    for _ in range(s+1,e):
        formatted_output[_] = 'I'+label[1:]

# 打印结果
for i in range(len(formatted_output)):
    print(example[i], formatted_output[i])