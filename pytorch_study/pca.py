import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 1. 加载数据
file_path = 'C:\\Users\\Administrator\\Downloads\\data_BS.csv'  # 替换为实际文件路径
data = pd.read_csv(file_path)

# 2. 数据分离
X = data.iloc[:, 1:]  # 假设第1列是标签
y = data.iloc[:, 0]   # 标签列

# 3. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 特征选择（选择100个重要特征，可根据需要调整）
selector = SelectKBest(score_func=f_classif, k=100)
X_selected = selector.fit_transform(X_scaled, y)

# 5. 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# 6. 构建MLP模型
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),  # 输入层和第一层
    Dropout(0.3),  # 防止过拟合
    Dense(64, activation='relu'),  # 第二层
    Dropout(0.3),
    Dense(32, activation='relu'),  # 第三层
    Dense(1, activation='sigmoid')  # 输出层，二分类
])

# 7. 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 8. 早停回调（防止过拟合）
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 9. 训练模型
history = model.fit(X_train, y_train, validation_split=0.2, epochs=500, batch_size=32, callbacks=[early_stopping], verbose=1)

# 10. 模型评估
y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

print("准确率:", accuracy_score(y_test, y_pred))
print("分类报告:\n", classification_report(y_test, y_pred))
print("ROC-AUC分数:", roc_auc_score(y_test, y_pred_prob))
