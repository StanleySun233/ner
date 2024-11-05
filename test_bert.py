import torch

# 定义输入参数
n_feature_maps = 128  # 特征图的数量，可以根据具体需求调整
n_classes = 10        # 类别数量
n_prototypes = 5      # 原型数量

# 创建 Dempster-Shafer 模型实例
model = Dempster_Shafer_module(n_feature_maps=n_feature_maps, n_classes=n_classes, n_prototypes=n_prototypes)

# 创建一个示例输入张量，假设批次大小为 32，特征图的数量为 n_feature_maps
batch_size = 32
inputs = torch.randn(batch_size, n_feature_maps)

# 前向传播，获取模型输出
outputs = model(inputs)

# 输出结果
print("模型输出：", outputs)
print("输出形状：", outputs.shape)  # 应该是 (batch_size, n_classes)，表示每个输入的类置信度
