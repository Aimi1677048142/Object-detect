import os
import random
import shutil

import cv2
import numpy as np

from ClassificationDateSet import ClassificationDateSet
from data_augmentation import rotate_image_and_boxes, flip_image, gaussian_sharpen, mean_sharpen, \
    adaptive_histogram_equalization


def create_subset(data_set, indices):
    """根据给定的索引创建一个新的 ClassificationDateSet 实例"""
    subset_data = [data_set.data[i] for i in indices]
    subset_labels = [data_set.labels[i] for i in indices]

    # 创建一个新的 ClassificationDateSet 实例
    subset = ClassificationDateSet(object_dir=data_set.object_dir)
    subset.data = subset_data
    subset.labels = subset_labels
    return subset


def random_sample_with_indices(data_set, num_train_samples, num_val_samples):
    """
    随机打乱数据并取指定数量的样本，返回两个 ClassificationDateSet 实例

    参数:
    data_set (ClassificationDateSet): 输入数据集
    num_train_samples (int): 要取的训练样本数量
    num_val_samples (int): 要取的验证样本数量

    返回:
    tuple: 包含两个 ClassificationDateSet 实例的元组
    """
    # 确保输入数据是 ClassificationDateSet 实例
    if not isinstance(data_set, ClassificationDateSet):
        raise TypeError("数据集必须是 ClassificationDateSet 的实例。")

    # 获取数据和标签
    data_list, labels = data_set.data, data_set.labels

    # 检查样本数量是否超过数据长度
    total_samples = num_train_samples + num_val_samples
    if total_samples > len(data_list):
        raise ValueError("总样本数量超过数据集的长度。")

    # 创建索引数组
    indices = np.arange(len(data_list))

    # 打乱数据和索引
    np.random.shuffle(indices)

    # 取指定数量的样本和对应的索引
    train_indices = indices[:num_train_samples]
    val_indices = indices[num_train_samples:num_train_samples + num_val_samples]

    # 创建训练和验证数据集
    train_set = create_subset(data_set, train_indices)
    val_set = create_subset(data_set, val_indices)

    return train_set, val_set


# 数据集的划分
def split_dataset(root_dir, train_ratio=0.8):
    # 创建训练集和验证集的目录
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    # 如果存在 'train' 和 'val' 目录，先删除再创建
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 用于保存类别名称和对应的数字编号
    class_mapping = {}
    subdirectory_count = 0  # 记录子目录数量

    # 遍历根目录中的所有子目录
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path) and class_name not in ['train', 'val']:
            # 保存类别名称和编号
            class_mapping[class_name] = subdirectory_count
            subdirectory_count += 1  # 增加子目录计数

            # 获取当前子目录中的所有文件
            files = os.listdir(class_path)
            random.shuffle(files)  # 打乱文件顺序

            # 计算训练集和验证集的分割点
            split_index = int(len(files) * train_ratio)
            train_files = files[:split_index]
            val_files = files[split_index:]

            # 创建子目录
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

            # 复制文件到训练集和验证集目录
            for file in train_files:
                shutil.copy(os.path.join(class_path, file), os.path.join(train_dir, class_name, file))
            for file in val_files:
                shutil.copy(os.path.join(class_path, file), os.path.join(val_dir, class_name, file))

    # 返回类别映射、创建的目录路径和子目录数量
    return class_mapping, train_dir, val_dir, subdirectory_count


def move_and_cleanup(root_dir):
    # 遍历根目录下的所有子目录
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)

        # 检查子目录是否是目录
        if os.path.isdir(subdir_path):
            train_dir = os.path.join(subdir_path, 'train')
            val_dir = os.path.join(subdir_path, 'val')

            # 如果有train文件夹，则移动其内容
            if os.path.exists(train_dir) and os.path.isdir(train_dir):
                for file_name in os.listdir(train_dir):
                    file_path = os.path.join(train_dir, file_name)
                    # 移动文件到子目录的根路径下
                    shutil.move(file_path, subdir_path)
                # 删除train文件夹
                shutil.rmtree(train_dir)
                print(f"Moved and removed 'train' from {subdir_path}")

            # 如果有val文件夹，则移动其内容
            if os.path.exists(val_dir) and os.path.isdir(val_dir):
                for file_name in os.listdir(val_dir):
                    file_path = os.path.join(val_dir, file_name)
                    # 移动文件到子目录的根路径下
                    shutil.move(file_path, subdir_path)
                # 删除val文件夹
                shutil.rmtree(val_dir)
                print(f"Moved and removed 'val' from {subdir_path}")


# root_dir = r'D:\pcb_data\DIB-10K-1_2_10_class1'  # 你的根目录路径
# move_and_cleanup(root_dir)


def merge_train_val(dataset_root):
    # 在上级目录中创建统一的 train 和 val 目录
    unified_train_dir = os.path.join(dataset_root, 'train')
    unified_val_dir = os.path.join(dataset_root, 'val')
    # 如果 train 和 val 目录已存在，先删除它们
    if os.path.exists(unified_train_dir):
        shutil.rmtree(unified_train_dir)
    if os.path.exists(unified_val_dir):
        shutil.rmtree(unified_val_dir)
    os.makedirs(unified_train_dir, exist_ok=True)
    os.makedirs(unified_val_dir, exist_ok=True)

    # 遍历 dataset_root 下的所有子目录
    for subdir in os.listdir(dataset_root):
        subdir_path = os.path.join(dataset_root, subdir)

        if os.path.isdir(subdir_path):
            # 查找 train 和 val 目录
            train_dir = os.path.join(subdir_path, 'train')
            val_dir = os.path.join(subdir_path, 'val')

            # 如果存在 train 目录，移动文件到统一的 train 目录
            if os.path.exists(train_dir):
                train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if
                               os.path.isfile(os.path.join(train_dir, f))]
                for file in train_files:
                    shutil.move(file, os.path.join(unified_train_dir, os.path.basename(file)))
                # 删除空的 train 目录
                shutil.rmtree(train_dir)

            # 如果存在 val 目录，移动文件到统一的 val 目录
            if os.path.exists(val_dir):
                val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if
                             os.path.isfile(os.path.join(val_dir, f))]
                for file in val_files:
                    shutil.move(file, os.path.join(unified_val_dir, os.path.basename(file)))
                # 删除空的 val 目录
                shutil.rmtree(val_dir)

    print(f"Files moved to unified directories: {unified_train_dir} and {unified_val_dir}")
    return unified_train_dir, unified_val_dir


# 使用示例
# dataset_root = r'D:\pcb_data\new_bird\new_bird'  # 替换为你的数据集根目录
# merge_train_val(dataset_root)


def split_dataset_for_sub_dirs(dataset_root, train_ratio=0.8, augment_ratio=0.5, flag=False):
    # 创建训练集和验证集的目录
    train_dir = os.path.join(dataset_root, 'train')
    val_dir = os.path.join(dataset_root, 'val')
    # 如果存在 'train' 和 'val' 目录，先删除再创建
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    class_mapping = {}  # 存储子目录名称到索引编号的映射
    subdirectory_count = 0  # 子目录数量计数
    # 遍历 dataset_root 下的所有子目录
    for index, subdir in enumerate(os.listdir(dataset_root)):
        if subdir in ['train', 'val']:
            continue
        subdir_path = os.path.join(dataset_root, subdir)

        if os.path.isdir(subdir_path):
            subdirectory_count += 1  # 增加子目录计数

            # 为当前子目录生成索引编号
            class_mapping[subdir] = index
            # 创建 train 和 val 目录
            train_dir = os.path.join(subdir_path, 'train')
            val_dir = os.path.join(subdir_path, 'val')

            # 如果 train 和 val 目录已存在，先删除它们
            if os.path.exists(train_dir):
                shutil.rmtree(train_dir)
            if os.path.exists(val_dir):
                shutil.rmtree(val_dir)

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            # 获取当前子目录中的所有文件，并过滤掉以 '.' 开头的隐藏文件夹和文件，如 .ipynb_checkpoints
            files = [file for file in os.listdir(subdir_path) if not file.startswith('.')]
            # 排除已经存在的 `train` 和 `val` 目录里的文件
            files = [file for file in files if file not in ['train', 'val']]
            random.shuffle(files)  # 随机打乱文件顺序

            # 计算训练集的大小
            train_size = int(len(files) * train_ratio)
            train_files = files[:train_size]
            val_files = files[train_size:]

            # 复制训练集文件到 train 子目录
            for file in train_files:
                src_path = os.path.join(subdir_path, file)
                dst_path = os.path.join(train_dir, file)
                shutil.copy(src_path, dst_path)
            # 复制验证集文件到 val 子目录
            for file in val_files:
                src_path = os.path.join(subdir_path, file)
                dst_path = os.path.join(val_dir, file)
                shutil.copy(src_path, dst_path)
            print(f"Completed split for {subdir}: {len(train_files)} files in train, {len(val_files)} files in val.")
    # 调用数据增强方法
    if flag:
        augment_dataset(dataset_root, augment_ratio)
    # 调用合并函数的方法
    unified_train_dir, unified_val_dir = merge_train_val(dataset_root)
    return class_mapping, unified_train_dir, unified_val_dir, subdirectory_count


# 示例用法
# dataset_root_directory = r"D:\three_classificate_data_set"  # 数据集的根目录
# print(split_dataset_for_sub_dirs(dataset_root_directory,flag=True))


def augment_dataset(root_dir, augment_ratio=0.5):
    """
    对训练集中的部分图像进行随机增强，并将增强后的图像保存到原目录中。

    :param root_dir: 训练集的根目录。
    :param augment_ratio: 增强的图像比例，默认为0.5。
    """
    # 遍历训练集中的所有子目录
    # aug_bird_list = [
    #     "1.Somali Ostrich",
    #     "42.Red-winged Tinamou",
    #     "82.Ross's Goose",
    #     "23.Little Tinamou",
    #     "26.Undulated Tinamou",
    #     "46.Chilean Tinamou",
    #     "58.Puna Tinamou",
    #     "11.Northern Cassowary",
    #     "45.Ornate Tinamou",
    #     "13.Grey Tinamou",
    #     "25.Brown Tinamou",
    #     "48.Andean Tinamou",
    #     "39.Small-billed Tinamou"
    # ]
    aug_bird_list = ["1.Barnacle Goose", "7.Fulvous Whistling Duck", "3.Canada Goose", "9.Black-bellied Whistling Duck",
                     "0.Common Ostrich"]
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if subdir not in aug_bird_list:
            continue
        if os.path.isdir(subdir_path):
            # 查找每个子目录下的train目录
            # train_dir = os.path.join(subdir_path, 'train')
            # if os.path.exists(train_dir) and os.path.isdir(train_dir):
            # 获取当前子目录中的所有图像文件
            image_files = [f for f in os.listdir(subdir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

            # 计算需要增强的图像数量
            augment_count = int(len(image_files) * augment_ratio)

            # 随机选择需要增强的图像
            selected_files = random.sample(image_files, augment_count)

            # 对每张选中的图像进行增强并保存
            for file in selected_files:
                img_path = os.path.join(subdir_path, file)
                img = cv2.imread(img_path)
                # 根据选择的图像数量，按比例进行旋转处理
                if selected_files.index(file) < augment_count * 0.3:  # 30% 的图像进行 90° 旋转
                    angle = 90
                elif selected_files.index(file) < augment_count * 0.6:  # 30% 的图像进行 180° 旋转
                    angle = 180
                else:  # 剩下的图像在 ±10° 到 ±30° 范围内旋转
                    angle = random.uniform(-30, 30)

                augmented_img, _ = rotate_image_and_boxes(img, None, angle, rotated_center_x=0.5,
                                                          rotated_center_y=0.5,
                                                          zoom=1.0, fill_up=False)
                new_filename = f"aug_{file}"
                cv2.imwrite(os.path.join(subdir_path, new_filename), augmented_img)
            # 自适应直方图
            adaptive_histogram_equalization_sample = random.sample(image_files, augment_count)
            for file in adaptive_histogram_equalization_sample:
                img_path = os.path.join(subdir_path, file)
                img = cv2.imread(img_path)
                equalization_img, _ = adaptive_histogram_equalization(img, None)
                new_filename = f'equalization_aug_{file}'
                cv2.imwrite(os.path.join(subdir_path, new_filename), equalization_img)
            # 高斯模糊
            gaussian_sharpen_sample = random.sample(image_files, augment_count)
            for file in gaussian_sharpen_sample:
                img_path = os.path.join(subdir_path, file)
                img = cv2.imread(img_path)
                gaussian_sharpen_img, _ = gaussian_sharpen(img, None, 0.4, 0.6)
                new_filename = f'gaussian_sharpen_aug_{file}'
                cv2.imwrite(os.path.join(subdir_path, new_filename), gaussian_sharpen_img)
            # 均值模糊
            mean_sharpen_sample = random.sample(image_files, augment_count)
            for file in mean_sharpen_sample:
                img_path = os.path.join(subdir_path, file)
                img = cv2.imread(img_path)
                mean_sharpen_img, _ = mean_sharpen(img, None, 0.4, 0.6)
                new_filename = f'mean_sharpen_aug_{file}'
                cv2.imwrite(os.path.join(subdir_path, new_filename), mean_sharpen_img)
            # 水平翻转
            flip_image_sample = random.sample(image_files, augment_count)
            for file in flip_image_sample:
                img_path = os.path.join(subdir_path, file)
                img = cv2.imread(img_path)
                flip_img, _ = flip_image(img)
                new_filename = f'flip_aug_{file}'
                cv2.imwrite(os.path.join(subdir_path, new_filename), flip_img)
            print(f"Completed augmentation for {subdir}: {augment_count} images augmented.")

# 使用示例
# train_directory = 'D:\pcb_data\original_ img and yolo_imfo'  # 替换为训练集的路径
# augment_dataset(train_directory, augment_ratio=0.5)
# if __name__ == "__main__":
# root_directory = r'D:\three_classificate_data_set'  # 请替换为你的数据集根目录
# class_mapping, train_directory, val_directory, subdirectory_count = split_dataset(root_directory)
# print("类别映射:", class_mapping)
# print("训练集目录:", train_directory)
# print("验证集目录:", val_directory)
# print("除了 train 和 val 的子目录数量:", subdirectory_count)
# augment_dataset(train_directory)
# 使用示例
# root_dir = r'D:\pcb_data\train_data'
# date_set = ClassificationDateSet(root_dir)
# print(len(date_set))
#
# # 随机抽样
# train_set, val_set = random_sample_with_indices(date_set, 2, 1)
# print("训练集大小:", len(train_set))
# print("验证集大小:", len(val_set))

#
# dataset_root_directory = r"D:\three_classificate_data_set"  # 数据集的根目录
# print(split_dataset_for_sub_dirs(dataset_root_directory,flag=True))
