import torch

def NAND(x1, x2):
    x = torch.tensor([x1, x2], dtype = torch.float32)
    w = torch.tensor([-0.5, -0.5], dtype = torch.float32)
    b = torch.tensor(0.7, dtype = torch.float32)
    output = torch.dot(w, x) + b
    return 1 if output >= 0 else 0

print(NAND(0, 0))
print(NAND(0, 1))
print(NAND(1, 0))
print(NAND(1, 1))