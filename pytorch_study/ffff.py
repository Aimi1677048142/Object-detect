# class GoogleNet(nn.Module):
#     def __init__(self, num_class, aux_flag=True, init_weights=False):
#         super(GoogleNet, self).__init__()
#         self.aux_flag = aux_flag
#         self.conv1 = Conv(3, 64, k=7, s=2, p=3)
#         self.max_pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
#
#         self.conv2 = Conv(64, 64, k=1)
#         self.conv3 = Conv(64, 192, k=3, s=1, p=1)
#         self.max_pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
#
#         self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
#         self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
#         self.max_pool3 = nn.MaxPool2d(3, 2, ceil_mode=True)
#
#         self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
#         self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
#         self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
#         # self.inception4d removed
#         self.inception4e = Inception(512, 256, 160, 320, 32, 128, 128)
#         self.max_pool4 = nn.MaxPool2d(3, 2, ceil_mode=True)
#
#         # self.inception5a removed
#         self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
#
#         if self.aux_flag:
#             self.aux1 = InceptionAux(512, 128, 2048, 1024, num_class)
#             self.aux2 = InceptionAux(528, 128, 2048, 1024, num_class)
#
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.dropout = nn.Dropout(0.4)
#         self.fc = nn.Linear(1024, num_class)
#
#         if init_weights:
#             self._initialize_weights()
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.max_pool1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.max_pool2(x)
#
#         x = self.inception3a(x)
#         x = self.inception3b(x)
#         x = self.max_pool3(x)
#
#         x = self.inception4a(x)
#         if self.training and self.aux_flag:
#             aux1 = self.aux1(x)
#
#         x = self.inception4b(x)
#         x = self.inception4c(x)
#
#         x = self.inception4e(x)
#         x = self.max_pool4(x)
#
#         x = self.inception5b(x)
#
#         x = self.avg_pool(x)
#         x = torch.flatten(x, 1)
#         x = self.dropout(x)
#         x = self.fc(x)
#
#         if self.training and self.aux_flag:
#             return x, aux2, aux1
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
