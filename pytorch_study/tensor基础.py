import torch


print(torch.cuda.is_available())

a = torch.tensor([[1,2,3],[4,5,6]],device="cpu",dtype=torch.float32) # 默认是CPU
b = torch.tensor([[1,2,3],[4,5,6]],device="cuda:0",dtype=torch.float32) # 电脑要支持gpu
print(a + b) # 不同device是不能计算的
a = torch.tensor([1,2,3])
b = torch.tensor([11,12,13])
x, y = torch.meshgrid(a,b)
#
# pass