import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. 读取Excel数据
file_path = "C:\\Users\\Administrator\\Downloads\\data_BS.csv"  # 替换为实际路径
data = pd.read_csv(file_path)

# 2. 数据预处理
# 假设A列是目标（0或1），B到最后一列是特征
target_column = "Mg_mean_0.01"  # A列
X = data.iloc[:, 1:].values  # B到最后一列是特征
y = data.iloc[:, 0].values   # A列是目标

# 标准化特征数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 逻辑回归模型
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 4. 模型预测
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

# 5. 评估模型
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"训练集精度: {train_accuracy:.4f}")
print(f"验证集精度: {val_accuracy:.4f}")
print("\n验证集分类报告:")
print(classification_report(y_val, y_val_pred))
