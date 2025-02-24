import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# 1. 数据加载与预处理
file_path = "C:\\Users\\Administrator\\Downloads\\data_BS.csv"  # 替换为你的数据路径
data = pd.read_csv(file_path)

# 假设最后一列是标签列，其他列是特征
X = data.iloc[:, 1:].values  # 前3250列是特征
y = data.iloc[:, 0].values  # 最后一列是标签

# 数据标准化（每列独立标准化，均值0，标准差1）
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 使用 PCA 进行降维
pca = PCA(n_components=100)  # 降维到 100 个主成分
X = pca.fit_transform(X)

# 数据划分（80%训练集，20%验证集）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# 2. 定义全连接神经网络
class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FullyConnectedNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 输入层 -> 隐藏层1
            nn.ReLU(),
            nn.Dropout(0.5),                  # Dropout 防止过拟合
            nn.Linear(hidden_dim, hidden_dim),  # 隐藏层1 -> 隐藏层2
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),         # 隐藏层 -> 输出层
            nn.Sigmoid()                      # Sigmoid 激活函数用于二分类
        )

    def forward(self, x):
        return self.model(x)

# 初始化模型参数
input_dim = X_train.shape[1]  # 输入维度
hidden_dim = 64               # 隐藏层维度

# 实例化模型
model = FullyConnectedNN(input_dim, hidden_dim)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 3. 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

# 将数据加载到 GPU 或 CPU
X_train, X_val = X_train.to(device), X_val.to(device)
y_train, y_val = y_train.to(device), y_val.to(device)

# 4. 训练模型
epochs = 500  # 训练轮数
batch_size = 32  # 每个小批次的样本数量

# 创建小批次数据加载函数
def get_batches(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i + batch_size], y[i:i + batch_size]

for epoch in range(epochs):
    # 训练模式
    model.train()
    train_loss = 0
    for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 验证模式
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        val_predictions = (val_outputs >= 0.5).float()
        val_accuracy = accuracy_score(y_val.cpu(), val_predictions.cpu())
        val_auc = roc_auc_score(y_val.cpu(), val_outputs.cpu())

    # 打印训练与验证信息
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}")

# 5. 验证结果
print("\nFinal Validation Results:")
val_outputs = model(X_val)
val_predictions = (val_outputs >= 0.5).float()
print(classification_report(y_val.cpu(), val_predictions.cpu()))
