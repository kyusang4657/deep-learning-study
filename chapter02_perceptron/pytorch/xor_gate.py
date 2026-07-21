import torch

def NAND(x1, x2):
    x = torch.tensor([x1, x2], dtype = torch.float32)
    w = torch.tensor([-0.5, -0.5], dtype = torch.float32)
    b = torch.tensor(0.7, dtype = torch.float32)
    output = torch.dot(w, x) + b
    return 1 if output >= 0 else 0

def OR(x1, x2):
    x = torch.tensor([x1, x2], dtype = torch.float32)
    w = torch.tensor([0.5, 0.5], dtype = torch.float32)
    b = torch.tensor(-0.2, dtype = torch.float32)
    output = torch.dot(w, x) + b
    return 1 if output >= 0 else 0

def AND(x1, x2):
    x = torch.tensor([x1, x2], dtype=torch.float32)
    w = torch.tensor([0.5, 0.5], dtype=torch.float32)
    b = torch.tensor(-0.7, dtype=torch.float32)
    tmp = torch.sum(w * x) + b
    return 1 if tmp > 0 else 0

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print(XOR(0, 0))
print(XOR(0, 1))
print(XOR(1, 0))
print(XOR(1, 1))