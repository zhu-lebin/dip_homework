import torch

try:
    a = torch.cuda.FloatTensor(2).zero_()
except Exception as e:
    print(e)
