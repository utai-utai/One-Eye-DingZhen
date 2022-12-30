import torch
from torch import nn
from data import word_list1, word_list2, dataset
from model import Network

# 调用cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("正在使用{}加速中……".format(device))

# 加载数据集
dataloader = dataset()

# 加载神经网络
network1 = Network(len(word_list1))
network2 = Network(len(word_list2))

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-2
optimizer1 = torch.optim.Adam(network1.parameters(), lr=learning_rate)
optimizer2 = torch.optim.Adam(network2.parameters(), lr=learning_rate)

# 记录训练次数 & 设置运行次数
train_step = 0
epoch = 20
for i in range(epoch):
    print("------第{}轮训练开始------".format(i + 1))
    # 训练开始
    network1.train()
    network2.train()
    for (img, target1, target2) in dataloader:
        # 第一个单词，计算loss
        outputs = network1(img)
        loss = loss_fn(outputs, target1)
        # 优化器调用
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()

        # 第二个单词，计算loss
        outputs = network2(img)
        loss = loss_fn(outputs, target2)
        # 优化器调用
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
    # 保存最终模型
    if i+1 == epoch:
        torch.save(network1, '../model/network_1.pth')
        torch.save(network2, '../model/network_2.pth')
        print('训练结束！模型保存成功！\n')
