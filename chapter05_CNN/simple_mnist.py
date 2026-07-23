import torch
import torch.nn as nn

# 28 * 28 MNIST 이미지를 처리하는 완전 연결 신경망
class SimpleDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
         x= self.flatten(x)
         x = self.fc1(x)
         x = self.relu(x)
         x = self.fc2(x)
         return x

# 모델 셍성 및 테스트
dnn_model = SimpleDNN()
print("SimpleDNN 모델 구조:")
print(dnn_model)

#테스트 입력
test_input = torch.randn(1, 1, 28, 28)
dnn_ouput = dnn_model(test_input)
print(f"DNN 출력 크기: {dnn_ouput.shape}")