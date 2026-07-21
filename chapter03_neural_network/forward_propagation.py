import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    # 모델의 구조를 준비
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)
    # 입력값의 계산 경로를 정함
    def forward(self, x):
         x= torch.relu(self.fc1(x))
         x = torch.sigmoid(self.fc2(x))
         return x # 최종 계산 결과 반환

model = SimpleNN()
sample_input = torch.tensor([1.0, 2.0])
output = model(sample_input)
print(f"SimpleNN 출력: {output}")