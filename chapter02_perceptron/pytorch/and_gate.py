import torch

def AND(x1, x2):
    x = torch.tensor([x1, x2], dtype=torch.float32)
    w = torch.tensor([0.5, 0.5], dtype=torch.float32)
    b = torch.tensor(-0.7, dtype=torch.float32)
    tmp = torch.sum(w * x) + b

    return 1 if tmp.item() <= 0 else 0

print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))
