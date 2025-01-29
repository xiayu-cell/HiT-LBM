import torch
b = torch.randn(5,4,10)
a = torch.randn(5,4,1).long()
print(a)

a = a/a.sum(dim=1).unsqueeze(-1)
print(a)

print((b*a).sum(dim=1).shape)