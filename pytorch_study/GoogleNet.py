from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import models
from tqdm import tqdm
import logging
from ClassificationCNN import ClassificationCNN
from ClassificationDateSet import ClassificationDateSet
from tool.base_tool import split_dataset

# 配置日志记录
logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')
class Conv(nn.Module):
    def __init__(self, in_c, out_c, k=1, s=1, p=0, d=1):
        """
        :param in_c: 输入的通道数
        :param out_c: 输出的通道数
        :param k: 卷积核
        :param s: 步长
        :param p: padding
        :param d: 膨胀
        """
        super(Conv, self).__init__()
        self.cov = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, dilation=d)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.cov(x)))


class Inception(nn.Module):
    def __init__(self, in_c, out_1x1, out_3x3_reduce, out_3x3, out_5x5_reduce, out_5x5, pool_proj):
        super(Inception, self).__init__()
        self.conv_1x1 = Conv(in_c, out_1x1, k=1)

        self.conv_3x3_reduce = Conv(in_c, out_3x3_reduce, k=1)
        self.conv_3x3 = Conv(out_3x3_reduce, out_3x3, k=3, s=1, p=1)

        self.conv_5x5_reduce = Conv(in_c, out_5x5_reduce, k=1)
        self.conv_5x5 = Conv(out_5x5_reduce, out_5x5, k=5, s=1, p=2)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_pool_proj = Conv(in_c, pool_proj, k=1)

    def forward(self, x):
        x1 = self.conv_1x1(x)
        x2 = self.conv_3x3(self.conv_3x3_reduce(x))
        x3 = self.conv_5x5(self.conv_5x5_reduce(x))
        x4 = self.conv_pool_proj(self.max_pool(x))
        return torch.concat([x1, x2, x3, x4], dim=1)


class InceptionAux(nn.Module):
    def __init__(self, in_c, outc, out_fc1, fc1_out, class_num):
        super(InceptionAux, self).__init__()
        self.average_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv_1x1 = Conv(in_c, outc, k=1, s=1)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(out_fc1, fc1_out)
        self.fc2 = nn.Linear(fc1_out, class_num)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv_1x1(self.average_pool(x))
        x = self.flat(x)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return x


class GoogleNet(nn.Module):
    def __init__(self, num_class, aux_flag=True, init_weights=False):
        super(GoogleNet, self).__init__()
        self.aux_flag = aux_flag
        self.conv1 = Conv(3, 64, k=7, s=2, p=3)
        self.max_pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.conv2 = Conv(64, 64, k=1)
        self.conv3 = Conv(64, 192, k=3, s=1, p=1)
        self.max_pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.max_pool3 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        # self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.max_pool4 = nn.MaxPool2d(3, 2, ceil_mode=True)

        # self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_flag:
            self.aux1 = InceptionAux(512, 128, 2048, 1024, num_class)
            self.aux2 = InceptionAux(528, 128, 2048, 1024, num_class)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_class)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):  # 网络的正向传播过程
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.max_pool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.max_pool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.max_pool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_flag:  # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        # x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_flag:  # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.max_pool4(x)
        # N x 832 x 7 x 7
        # x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avg_pool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_flag:  # eval model lose this layer
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    root_dir = r'D:\three_classificate_data_set'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 数据集的划分
    class_mapping, train_dir, val_dir, subdirectory_count = split_dataset(root_dir)
    print("类别映射：", class_mapping)
    # 使用字典推导式来交换键和值
    inverted_class_mapping = {v: k for k, v in class_mapping.items()}
    train_date_set = ClassificationDateSet(train_dir, class_mapping)
    val_date_set = ClassificationDateSet(val_dir, class_mapping)
    model = GoogleNet(subdirectory_count, aux_flag=True, init_weights=True).to(device)

    batch_size, learning_rate, epochs = 64, 1e-4, 1
    # 测试数据集
    train_loader = DataLoader(
        dataset=train_date_set,
        batch_size=batch_size,
        shuffle=True
    )
    # 验证数据集
    val_loader = DataLoader(
        dataset=val_date_set,
        batch_size=batch_size,
        shuffle=True
    )
    # 设置优化器,学习率
    optim_adam = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 设置损失函数
    criterion = nn.CrossEntropyLoss().to(device)


    def model_train(data, label, epoch):
        # 清空模型的梯度
        model.zero_grad()
        # 前向传播 (返回主输出和两个辅助输出)
        out_puts, aux1, aux2 = model(data)

        # 计算主输出的损失
        loss_main = criterion(out_puts, label)
        # 计算辅助输出的损失
        loss_aux1 = criterion(aux1, label)
        loss_aux2 = criterion(aux2, label)

        # 合并总损失，给辅助分类器损失加权
        loss = loss_main + 0.3 * (loss_aux1 + loss_aux2)
        # 计算损失
        # 推断
        predict = torch.argmax(out_puts, dim=1)
        # 计算精度
        accuracy = torch.sum(predict == label) / label.shape[0]
        # 反向传播
        loss.backward()
        # 权重更新
        optim_adam.step()
        # 进度条显示loss和acc
        desc = "train.[{}/{}] loss:{:.4f}, Acc:{:.4f}".format(epoch, epochs, loss.item(), accuracy.item())
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
        return loss, predict_num


    # 初始化每个类别的计数器
    class_counts = defaultdict(int)
    correct_counts = defaultdict(int)
    for epoch in range(1, epochs + 1):
        process_bar = tqdm(train_loader, unit='step')
        # 开始训练
        model.train(True)
        train_loss, train_correct = 0, 0
        for step, (train_data, train_label) in enumerate(process_bar):
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
                result_num=0
                for _, (val_data, label) in enumerate(val_loader):
                    # label = nn.functional.one_hot(label, 10)
                    val_data, label = val_data.to(device), label.to(device)  # 移动到 GPU
                    val_loss, val_predict_num = model_val(val_data, label)
                    total_loss += val_loss
                    total_correct += val_predict_num
                    result_num+=len(val_data)
                    # 更新验证集的类别计数
                    for val_label in label:
                        class_counts[val_label.item()] += 1
                    # 更新验证集的正确预测计数
                    val_predict = torch.argmax(model(val_data), dim=1)
                    for val_label, val_pred in zip(label, val_predict):
                        if val_label == val_pred:
                            correct_counts[val_label.item()] += 1
                val_total_mean_correct = total_correct / result_num
                val_total_mean_loss = total_loss / len(val_loader)
                train_total_loss = train_loss / len(train_loader)
                train_total_correct = train_correct / len(process_bar)
                # 进度条显示loss和acc
                train_desc = "train.[{}/{}] loss:{:.4f}, Acc:{:.4f}".format(epoch, epochs, train_total_loss.item(),
                                                                            train_total_correct.item())
                # 验证集的日志
                val_desc = "val.[{}/{}] loss:{:.4f}, Acc:{:.4f}".format(epoch, epochs, val_total_mean_loss.item(),
                                                                        val_total_mean_correct.item())
                process_bar.set_description(train_desc + val_desc)

        # 打印每个类别的精度
        # 记录每个类别的精度及其相关信息
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
        torch.save(model, './torch_mnist2.pt')
# GoogleNet
