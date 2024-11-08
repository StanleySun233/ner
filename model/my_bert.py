import torch
from mega_pytorch import MegaLayer
from torch import nn
from torchcrf import CRF
from transformers import AutoModelForTokenClassification, MambaModel
from transformers import MambaPreTrainedModel

from model import dst


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
        torch.nn.init.normal_(self.crf.transitions, mean=0, std=0.1)


class BertBiLSTMMegaCRF(AutoModelForTokenClassification):
    def __init__(self, config, lstm_hidden_size=256):
        super().__init__(config)

        # BiLSTM 层
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(lstm_hidden_size * 2)

        # MegaLayer 层
        self.mega = MegaLayer(
            dim=lstm_hidden_size,
            ema_heads=32,
            attn_dim_qk=128,
            attn_dim_value=256,
            laplacian_attn_fn=False,
        )

        # 线性分类器
        self.classifier = nn.Linear(lstm_hidden_size * 2, config.num_labels)

        # CRF 层
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 获取 BERT 的输出
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
        sequence_output = self.dropout(sequence_output)
        # 第一残差块：BiLSTM 层 + 残差
        lstm_output, _ = self.lstm(sequence_output)  # (batch_size, seq_length, lstm_hidden_size * 2)
        lstm_output = lstm_output + sequence_output  # 将输入添加到 LSTM 输出
        lstm_output = self.layer_norm(lstm_output)

        # 第二残差块：MegaLayer 层 + 残差
        mega_output = self.mega(lstm_output) + lstm_output

        # 第三残差块：分类器 + 残差
        logits = self.classifier(mega_output)

        # 计算损失或解码预测
        if labels is not None:
            # 计算 CRF 损失
            log_likelihood = self.crf(logits, labels, mask=attention_mask.byte())
            return -log_likelihood * 10  # 返回负对数似然作为损失
        else:
            # 解码预测标签
            predictions = self.crf.decode(logits, mask=attention_mask.byte())
            return predictions


class BertMegaBiLSTMCRF(AutoModelForTokenClassification):
    def __init__(self, config, lstm_hidden_size=256):
        super().__init__(config)

        # MegaLayer 层
        self.mega = MegaLayer(
            dim=config.hidden_size,
            ema_heads=32,
            attn_dim_qk=128,
            attn_dim_value=256,
            laplacian_attn_fn=False,
        )

        # BiLSTM 层
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(lstm_hidden_size * 2)

        # 线性分类器
        self.classifier = nn.Linear(lstm_hidden_size * 2, config.num_labels)

        # CRF 层
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 获取 BERT 的输出
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
        # sequence_output = self.dropout(sequence_output)

        # 第一残差块：MegaLayer 层 + 残差
        mega_output = self.mega(sequence_output)

        # 第二残差块：BiLSTM 层 + 残差
        lstm_output, _ = self.lstm(mega_output)  # (batch_size, seq_length, lstm_hidden_size * 2)
        lstm_output = lstm_output + mega_output  # 将 MegaLayer 输出添加到 LSTM 输出
        lstm_output = self.layer_norm(lstm_output)

        # 第三残差块：分类器 + 残差
        logits = self.classifier(lstm_output)

        # 计算损失或解码预测
        if labels is not None:
            # 计算 CRF 损失
            log_likelihood = self.crf(logits, labels, mask=attention_mask.byte())
            return -log_likelihood * 10  # 返回负对数似然作为损失
        else:
            # 解码预测标签
            predictions = self.crf.decode(logits, mask=attention_mask.byte())
            return predictions

class BertMegaCRF(AutoModelForTokenClassification):
    def __init__(self, config, lstm_hidden_size=256):
        super().__init__(config)

        # MegaLayer 层
        self.mega = MegaLayer(
            dim=config.hidden_size,
            ema_heads=32,
            attn_dim_qk=128,
            attn_dim_value=256,
            laplacian_attn_fn=False,
        )

        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(lstm_hidden_size * 2)

        # 线性分类器
        self.classifier = nn.Linear(lstm_hidden_size * 2, config.num_labels)

        # CRF 层
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 获取 BERT 的输出
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
        sequence_output = self.dropout(sequence_output)

        # 第一残差块：MegaLayer 层 + 残差
        mega_output = self.mega(sequence_output)

        mega_output = self.layer_norm(mega_output)

        # 第三残差块：分类器 + 残差
        logits = self.classifier(mega_output)

        # 计算损失或解码预测
        if labels is not None:
            # 计算 CRF 损失
            log_likelihood = self.crf(logits, labels, mask=attention_mask.byte())
            return -log_likelihood * 10  # 返回负对数似然作为损失
        else:
            # 解码预测标签
            predictions = self.crf.decode(logits, mask=attention_mask.byte())
            return predictions
# class BertBiLSTMDSTCRF(AutoModelForTokenClassification):
#     def __init__(self, config, lstm_hidden_size=256):
#         super().__init__(config)
#         self.lstm = nn.LSTM(
#             input_size=config.hidden_size,
#             hidden_size=lstm_hidden_size,
#             num_layers=1,
#             bidirectional=True,
#             batch_first=True
#         )
#         self.dst = dst.Dempster_Shafer_module(lstm_hidden_size * 2, config.num_labels, 5)
#         self.classifier = nn.Linear(lstm_hidden_size * 2, config.num_labels)
#         self.crf = CRF(config.num_labels, batch_first=True)
#
#     def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
#         # 获取 BERT 的输出
#         outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs)
#         sequence_output = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
#
#         # 通过 BiLSTM 层
#         lstm_output, _ = self.lstm(sequence_output)  # (batch_size, seq_length, lstm_hidden_size * 2)
#
#         # 通过线性分类器，将 lstm_output 的输出映射到标签空间
#         logits = self.dst(lstm_output)  # (batch_size, seq_length, num_labels)
#
#         # 计算损失或解码预测
#         if labels is not None:
#             # 计算 CRF 损失
#             log_likelihood = self.crf(logits, labels, mask=attention_mask.byte())
#             return -log_likelihood  # 返回负对数似然作为损失
#         else:
#             # 解码预测标签
#             predictions = self.crf.decode(logits, mask=attention_mask.byte())
#             return predictions
#
#
# class MambaBiLSTMCRF(MambaPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         lstm_hidden_size = 256
#
#         # 使用 Mamba 模型作为编码器
#         self.mamba = MambaModel.from_pretrained("state-spaces/mamba-130m-hf", config=config)
#         self.num_labels = config.num_labels
#
#         # 添加 BiLSTM 层
#         self.lstm = nn.LSTM(
#             input_size=config.hidden_size,
#             hidden_size=lstm_hidden_size,
#             num_layers=1,
#             bidirectional=True,
#             batch_first=True
#         )
#
#         # 更新线性层的输入大小以适应 BiLSTM 的输出
#         self.classifier = nn.Linear(lstm_hidden_size * 2, config.num_labels)
#
#         # 添加 CRF 层
#         self.crf = CRF(config.num_labels, batch_first=True)
#
#     def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
#         # 获取 Mamba 模型的输出
#         outputs = self.mamba(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
#         sequence_output = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
#
#         # 通过 BiLSTM 层
#         lstm_output, _ = self.lstm(sequence_output)  # (batch_size, seq_length, lstm_hidden_size * 2)
#
#         # 通过线性分类器，将 lstm_output 的输出映射到标签空间
#         logits = self.classifier(lstm_output)  # (batch_size, seq_length, num_labels)
#         print("Logits shape:", logits.shape)  # 应该是 (batch_size, sequence_length, num_labels)
#         print("Labels shape:", labels.shape)  # 应该是 (batch_size, sequence_length)
#
#         # 计算损失或解码预测
#         if labels is not None:
#             # 计算 CRF 损失
#             log_likelihood = self.crf(logits, labels, mask=attention_mask.byte())
#             return -log_likelihood  # 返回负对数似然作为损失
#         else:
#             # 解码预测标签
#             predictions = self.crf.decode(logits, mask=attention_mask.byte())
#             return predictions
