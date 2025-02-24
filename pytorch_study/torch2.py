import torch
from torch.autograd import Variable
# X
scalar = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
# 创建权重参数层 W
var = Variable(scalar, requires_grad=True)
# 损失函数
v_out = torch.mean(var * var-3*var)
# 求导
v_out.backward()
# 得到求导的值
print(var.grad)

"""
    net = GoogleNet(classes=5,isTrain=True)

    for x,y in dataloader:
        y1,y2,y3 = net(x)
        l0 = loss(y1,y)
        l1 = loss(y2,y)
        l2 = loss(y3,y)
        total_loss = l0 + 0.3 * l1 + 0.3 * l2
        total_loss.back....()
        total_loss.step()

"""
