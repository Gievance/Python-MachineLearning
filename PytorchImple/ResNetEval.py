# Writer：TuTTTTT
# 编写Time:2024/3/20 20:42
# Writer：TuTTTTT
# 编写Time:2024/3/16 10:05
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PytorchImple.Models.ResNet import ResNet, BasicBlock, BottleNeck
from tqdm import tqdm

# 1. 定义数据加载器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 返回训练集迭代对象

test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # 返回测试机迭代对象

# 2. 定义模型
layer=[3,4,6,3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(block=BottleNeck, layer=layer).to(device)

# 3. 定义损失函数
criterion = nn.CrossEntropyLoss()

# 4. 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 训练模型
epochs = 4
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    progress_bar = tqdm(train_loader, total=len(train_loader),
                        desc=f'Epoch {epoch + 1}/{epochs}')  # tqdm会返回train_loader的迭代器
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # 梯度清零
        outputs = model(inputs)
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 误差反向传播
        optimizer.step()  # 参数更新

        running_loss += loss.item()
        progress_bar.set_postfix({'Training Loss': running_loss / len(train_loader)})  # 更新进度条显示的训练损失
# 6. 保存模型
torch.save(model.state_dict(), 'Weights/densenet_cifar10.pth')

# 7. 加载训练好的模型进行推理
model = ResNet(block=BottleNeck,layer=layer).to(device)
model.load_state_dict(torch.load('Weights/densenet_cifar10.pth'))

# 8. 模型评估
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {(100 * correct / total):.2f}%")
