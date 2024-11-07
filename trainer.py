import utils
import evaluate
from model.my_bert import BertBiLSTMMegaCRF
from huggingface_hub import notebook_login

dataset_name = "PassbyGrocer/weibo-ner"
pretrained_name = "hfl/chinese-roberta-wwm-ext-large"
user_id = "PassbyGrocer"
model_name = "robert_bilstm_mega_res"
model_cls = BertBiLSTMMegaCRF

print(utils.check_torch_gpu())

tokenized_datasets, data_collator,tokenizer = utils.load_datasets_by_hf(dataset_name,pretrained_name)
model = utils.load_model_by_hf(pretrained_name,tokenized_datasets,model_cls)
metric = evaluate.load("seqeval")

label_list = tokenized_datasets["train"].features["ner_tags"].feature.names  # 获取标签名称列表
s = [i for i in range(len(label_list))]

def compute_metrics_helper(pred):
    return utils.compute_metrics(pred, label_list,metric)

train_args = utils.get_train_args(user_id,model_name,dataset_name.split("/")[1])
trainer = utils.get_trainer(model,train_args,tokenized_datasets,tokenizer,data_collator)

trainer.train()

test_results = trainer.predict(tokenized_datasets["test"])
# 输出测试结果
print("Result in Test:", test_results.metrics)

trainer.push_to_hub()