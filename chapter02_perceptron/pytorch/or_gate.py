import torch

def OR(x1, x2):
    x = torch.tensor([x1, x2], dtype = torch.float32)
    w = torch.tensor([0.5, 0.5], dtype = torch.float32)
    b = torch.tensor(-0.2, dtype = torch.float32)
    output = torch.dot(w, x) + b
    return 1 if output >= 0 else 0

print(OR(0, 0))
print(OR(0, 1))
print(OR(1, 0))
print(OR(1, 1))