import torch
from transformers import AutoModelForTokenClassification, MambaModel, MambaConfig
from torchcrf import CRF
from torch import nn
from transformers import MambaPreTrainedModel
class AutoModelForTokenClassificationWithCRF(AutoModelForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 获取原始 BERT 的输出 logits
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs)
        logits = outputs.logits  # (batch_size, seq_length, num_labels)

        # 计算损失或解码预测
        if labels is not None:
            # 计算 CRF 损失
            log_likelihood = self.crf(logits, labels, mask=attention_mask.byte())
            return -log_likelihood  # 返回负对数似然作为损失
        else:
            # 解码预测标签
            predictions = self.crf.decode(logits, mask=attention_mask.byte())
            return predictions


class BertBiLSTMCRF(AutoModelForTokenClassification):
    def __init__(self, config, lstm_hidden_size=256):
        super().__init__(config)

        # 添加 BiLSTM 层，输入大小为 BERT 的隐藏层大小，输出大小为自定义的 lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # 更新线性层的输入大小以适应 BiLSTM 的输出
        self.classifier = nn.Linear(lstm_hidden_size * 2, config.num_labels)

        # 添加 CRF 层
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 获取 BERT 的输出
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)

        # 通过 BiLSTM 层
        lstm_output, _ = self.lstm(sequence_output)  # (batch_size, seq_length, lstm_hidden_size * 2)

        # 通过线性分类器，将 lstm_output 的输出映射到标签空间
        logits = self.classifier(lstm_output)  # (batch_size, seq_length, num_labels)

        # 计算损失或解码预测
        if labels is not None:
            # 计算 CRF 损失
            log_likelihood = self.crf(logits, labels, mask=attention_mask.byte())
            return -log_likelihood  # 返回负对数似然作为损失
        else:
            # 解码预测标签
            predictions = self.crf.decode(logits, mask=attention_mask.byte())
            return predictions

class MambaBiLSTMCRF(MambaPreTrainedModel):
    def __init__(self,model_name, config:MambaConfig, lstm_hidden_size=256):
        super().__init__(config)
        # 使用 Mamba 模型作为编码器
        self.mamba = MambaModel.from_pretrained(model_name,config)
        # 添加 BiLSTM 层，输入大小为 Mamba 的隐藏层大小，输出大小为自定义的 lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # 更新线性层的输入大小以适应 BiLSTM 的输出
        self.classifier = nn.Linear(lstm_hidden_size * 2, config.num_labels)

        # 添加 CRF 层
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 获取 Mamba 模型的输出
        outputs = self.mamba(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)

        # 通过 BiLSTM 层
        lstm_output, _ = self.lstm(sequence_output)  # (batch_size, seq_length, lstm_hidden_size * 2)

        # 通过线性分类器，将 lstm_output 的输出映射到标签空间
        logits = self.classifier(lstm_output)  # (batch_size, seq_length, num_labels)

        # 计算损失或解码预测
        if labels is not None:
            # 计算 CRF 损失
            log_likelihood = self.crf(logits, labels, mask=attention_mask.byte())
            return -log_likelihood  # 返回负对数似然作为损失
        else:
            # 解码预测标签
            predictions = self.crf.decode(logits, mask=attention_mask.byte())
            return predictions

