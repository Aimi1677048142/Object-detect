import numpy as np
import torch

array_list = np.array([[1, 2], [3, 4]])
torch_from_numpy = torch.from_numpy(array_list)
torch_zeros = torch.zeros([3, 3])
torch_ones = torch.ones([3, 3])
torch_empty = torch.empty([3, 3])  # 只分配内存，但是数据是随机值
torch_full = torch.full([3, 3], 2)
torch_eye = torch.eye(2)
torch_arange = torch.arange(1, 10, 2)
torch_linspace = torch.linspace(1, 10, 10)
torch_logspace = torch.logspace(1, 100, 10)
torch_normal = torch.normal(torch.tensor(1.0), torch.tensor(1e-6))
torch_rand = torch.rand([3, 3])
torch_chunk = torch.chunk(torch_from_numpy, 2)
index = torch.tensor([[0],[1]])
# torch_gather = torch.gather(torch_from_numpy, dim=1, index=index)
# torch_index_select = torch.index_select(torch_from_numpy, dim=1, index=index)
masked_select = torch.masked_select(torch_from_numpy, torch_from_numpy > 2)
torch_narrow = torch.narrow(torch_from_numpy, 1,0,1)
torch_nonzero = torch.nonzero(torch_from_numpy)
print(torch_nonzero)
