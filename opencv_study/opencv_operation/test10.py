import cv2
import numpy as np

# 假设我们有已知的原始坐标
src_pts = np.float32([[0, 0], [300, 0], [300, 400], [0, 400]])

# 透视变换角度：比如你知道图像旋转或变换角度，创建变换矩阵
# 比如目标是通过透视变换，把一个正方形变成倾斜的梯形
angle = 30  # 假设30度角
alpha = np.deg2rad(angle)

# 假设我们只知道变换角度（例如从一个正面视角转变到一个倾斜视角）
dst_pts = np.float32([[0, 0], [300, 0], [300 - 100 * np.cos(alpha), 400], [100 * np.sin(alpha), 400]])

# 计算透视变换矩阵
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# 打印变换矩阵
print("透视变换矩阵:", M)

# 如果你要推导目标坐标, 你可以使用 getPerspectiveTransform 来获得反向矩阵
M_inv = np.linalg.inv(M)

# 应用到原始坐标推导目标点
points = np.float32([[75, 75], [150, 150]]).reshape(-1, 1, 2)
transformed_points = cv2.perspectiveTransform(points, M_inv)

print("原始点:", points)
print("逆推的变换后的目标点:", transformed_points)
