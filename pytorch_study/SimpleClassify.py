import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
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
pca = PCA(n_components=0.99)  # 降维到 100 个主成分
X = pca.fit_transform(X)

# 数据划分（80%训练集，20%验证集）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. 定义 SVM 模型
svm_model = SVC(kernel="rbf", C=1.0, probability=True, random_state=42,class_weight="balanced")  # 使用 RBF 核函数

# 3. 训练模型
svm_model.fit(X_train, y_train)

# 4. 验证模型
y_val_pred = svm_model.predict(X_val)
y_val_prob = svm_model.predict_proba(X_val)[:, 1]  # 获取预测概率

# 计算验证集精度
val_accuracy = accuracy_score(y_val, y_val_pred)

# 计算 AUC
val_auc = roc_auc_score(y_val, y_val_prob)

# 打印结果
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation AUC: {val_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

# 5. 打印 PCA 信息（可选）
explained_variance_ratio = pca.explained_variance_ratio_
print(f"\nExplained Variance Ratio of PCA: {explained_variance_ratio[:10]}")  # 打印前10个主成分的方差贡献率
