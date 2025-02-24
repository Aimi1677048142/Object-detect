import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import shutil
from ClassificationDateSet import ClassificationDateSet
from tool.base_tool import split_dataset, split_dataset_for_sub_dirs, augment_dataset

# 配置日志记录
logging.basicConfig(filename='vgg_resnet_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入和输出的通道数不同，需要用1x1卷积调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)  # 直接连接
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # 残差连接
        out = self.relu(out)
        return out


class ClassificationCNN(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationCNN, self).__init__()
        # VGG-like features
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Add ResNet block after VGG features
        # Add multiple ResNet blocks after VGG features
        self.resnet_blocks = nn.Sequential(
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.resnet_blocks(x)  # Apply ResNet block
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 运行测试
if __name__ == "__main__":
    root_dir = r'/root/autodl-tmp/DIB-10K-1_2_10_class1'
    # 定义目录名称
    annotated_images_dir = '/root/autodl-tmp/annotated_images'
    # 删除目录及其内容（如果存在）
    if os.path.exists(annotated_images_dir):
        shutil.rmtree(annotated_images_dir)
    # 创建目录
    os.makedirs(annotated_images_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 数据集的划分
    class_mapping, train_dir, val_dir, subdirectory_count = split_dataset(root_dir)
    logging.info(f"类别映射：{class_mapping}")
    # augment_dataset(train_dir)
    # 使用字典推导式来交换键和值
    inverted_class_mapping = {v: k for k, v in class_mapping.items()}
    train_date_set = ClassificationDateSet(train_dir, class_mapping)
    val_date_set = ClassificationDateSet(val_dir, class_mapping)
    model = ClassificationCNN(subdirectory_count).to(device)

    batch_size, learning_rate, epochs = 16, 1e-4, 50
    # 测试数据集
    train_loader = DataLoader(
        dataset=train_date_set,
        batch_size=batch_size,
        shuffle=True, num_workers=8
    )
    # 验证数据集
    val_loader = DataLoader(
        dataset=val_date_set,
        batch_size=batch_size,
        shuffle=True, num_workers=8
    )
    # 设置优化器,学习率
    # 设置衰退率
    weight_decay = 1e-4
    optim_adam = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 设置损失函数
    criterion = nn.CrossEntropyLoss().to(device)


    def model_train(data, label, epoch):
        # 清空模型的梯度
        model.zero_grad()
        # 前向传播
        out_puts = model(data)
        # 计算损失
        loss = criterion(out_puts, label)
        # 推断
        predict = torch.argmax(out_puts, dim=1)
        # 计算精度
        accuracy = torch.sum(predict == label) / label.shape[0]
        # 反向传播
        loss.backward()
        # 权重更新
        optim_adam.step()
        # 进度条显示loss和acc
        desc = "train.[{}/{}] loss:{:.4f}, Acc:{:.2f}".format(epoch, epochs, loss, accuracy)
        return loss, accuracy, desc


    @torch.no_grad
    def model_val(data, val_label):
        # 前向传播
        out_puts = model(data)
        # 计算损失
        loss = criterion(out_puts, val_label)
        # 计算前向传播结果与标签一致的数量
        val_puts = torch.argmax(out_puts, dim=1)
        predict_num = torch.sum(val_puts == val_label)
        return loss, predict_num, val_puts


    # 初始化训练集/验证集列表
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    # 初始化每个类别的计数器

    # all_val_images = []  # 用于存储验证集的图像数据
    # all_val_labels = []  # 用于存储验证集的真实标签
    # all_val_predictions = []  # 用于存储预测标签
    # all_val_filenames = []
    for epoch in range(1, epochs + 1):
        class_counts = defaultdict(int)
        correct_counts = defaultdict(int)
        process_bar = tqdm(train_loader, unit='step')
        # 开始训练
        model.train(True)
        train_loss, train_correct = 0, 0
        for step, (train_data, train_label, train_filename) in enumerate(process_bar):
            # 调用训练函数进行训练
            # train_label = nn.functional.one_hot(train_label,10)
            # 移动到 GPU
            train_data, train_label = train_data.to(device), train_label.to(device)
            loss, accuracy, desc = model_train(train_data, train_label, epoch)
            train_loss += loss
            train_correct += accuracy

            # 输出日志
            process_bar.set_description(desc)
            if step == len(process_bar) - 1:
                total_loss, total_correct = 0, 0
                model.eval()
                result_num = 0
                with torch.no_grad():
                    for _, (val_data, label, val_filename) in enumerate(val_loader):
                        # label = nn.functional.one_hot(label, 10)
                        val_data, label = val_data.to(device), label.to(device)  # 移动到 GPU
                        val_filename = val_filename
                        val_loss, val_predict_num, val_predict = model_val(val_data, label)
                        total_loss += val_loss
                        total_correct += val_predict_num
                        result_num += len(val_data)
                        # 存储当前批次的验证数据、标签和预测结果
                        # all_val_images.append(val_data.cpu())
                        # all_val_labels.append(label.cpu())
                        # all_val_predictions.append(val_predict.cpu())
                        # all_val_filenames.append(val_filename)
                        # 更新验证集的类别计数
                        for val_label in label:
                            class_counts[val_label.item()] += 1
                        # 更新验证集的正确预测计数
                        val_predict = torch.argmax(model(val_data), dim=1)
                        for val_label, val_pred in zip(label, val_predict):
                            if val_label == val_pred:
                                correct_counts[val_label.item()] += 1
                        # 逐批绘制和保存图像
                        if epoch == epochs:
                            for i in range(len(val_data)):
                                img = val_data[i].cpu().numpy().transpose(1, 2, 0)  # 转换为HWC格式
                                true_label = label[i].item()
                                pred_label = val_predict[i].item()
                                img_filename = val_filename[i]
                                plt.imshow(img)
                                plt.title(
                                    f"True: {inverted_class_mapping[true_label]}, Pred: {inverted_class_mapping[pred_label]}")
                                plt.text(0.5, -0.1, f"Filename: {img_filename}", ha='center', va='center',
                                         transform=plt.gca().transAxes, fontsize=10)
                                plt.axis('off')
                                plt.savefig(os.path.join(annotated_images_dir, f"epoch{epoch}_batch{step}_img{i}.png"))
                                plt.close()  # 关闭当前图像，释放内存

                        # 清理无用变量
                        del val_data, label
                        torch.cuda.empty_cache()
                # print("计算总数：", result_num)
                # print("总正确数：", total_correct)
                val_total_mean_correct = total_correct / result_num
                val_total_mean_loss = total_loss / len(val_loader)
                train_total_loss = train_loss / len(train_loader)
                train_total_correct = train_correct / len(process_bar)

                # 添加数据到图片
                train_losses.append(train_total_loss.item())
                val_losses.append(val_total_mean_loss.item())
                train_accuracies.append(train_total_correct.item())
                val_accuracies.append(val_total_mean_correct.item())
                # 进度条显示loss和acc
                train_desc = "train.[{}/{}] loss:{:.4f}, Acc:{:.4f}".format(epoch, epochs, train_total_loss.item(),
                                                                            train_total_correct.item())
                # 验证集的日志
                val_desc = "val.[{}/{}] loss:{:.4f}, Acc:{:.4f}".format(epoch, epochs, val_total_mean_loss.item(),
                                                                        val_total_mean_correct.item())
                process_bar.set_description(train_desc + val_desc)
        # 打印每个类别的精度
        # 记录每个类别的精度及其相关信息
        if epoch == epochs:
            logging.info(f"Epoch {epoch}:")
            logging.info("每个类别的精度和统计：")
            for class_id in class_counts:
                total_class_samples = class_counts[class_id]
                correct_class_samples = correct_counts[class_id]
                accuracy = correct_class_samples / total_class_samples if total_class_samples > 0 else 0
                category = inverted_class_mapping.get(class_id, f"类别 {class_id}")
                logging.info(
                    f"类别 {category}: 样本总数 {total_class_samples}, 正确数 {correct_class_samples}, 精度 {accuracy:.4f}")
        process_bar.close()
        # 清理并释放内存
        del train_data, train_label
        torch.cuda.empty_cache()
    torch.save(model, './cnn_resnet_torch_mnist2.pt')

    # Plotting the loss and accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    # 保存图像到文件
    plt.savefig('bird_training_validation_metrics.png')  # 你可以指定任何你想要的文件名和格式
    plt.show()

    # for batch_idx in range(len(all_val_images)):
    #     val_images = all_val_images[batch_idx]
    #     val_labels = all_val_labels[batch_idx]
    #     val_predictions = all_val_predictions[batch_idx]
    #     val_filenames = all_val_filenames[batch_idx]
    #     for i in range(len(val_images)):
    #         img = val_images[i].numpy().transpose(1, 2, 0)  # 转换为HWC格式以便可视化
    #         true_label = val_labels[i].item()
    #         pred_label = val_predictions[i].item()
    #         img_filename = val_filenames[i]
    #         # 绘制并保存图像
    #         plt.imshow(img)
    #         plt.title(f"True: {inverted_class_mapping[true_label]}, Pred: {inverted_class_mapping[pred_label]}")
    #         # 在图像下方添加图片的文件名称作为标签
    #         plt.text(0.5, -0.1, f"Filename: {img_filename}", ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    #         plt.axis('off')
    #         # **保存所有图像在一个目录下，并确保文件名唯一**
    #         plt.savefig(os.path.join(annotated_images_dir, f"epoch{epoch}_batch{batch_idx}_img{i}.png"))
    #         plt.close()
