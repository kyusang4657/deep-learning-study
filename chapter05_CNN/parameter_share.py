import torch
import torch.nn.functional as F

# (필터 개수, 입력 채널 수, 필터 높이, 필터 너비)
sobel_filter = torch.tensor([[[
    [1.0, 0.0, -1.0],
    [2.0, 0.0, -2.0],
    [1.0, 0.0, -1.0]
]]])

# 이미지의 여러 위치에 동일한 필터 적용 (Sobel 필터의 예)
sample_28x28 = torch.randn(1, 1, 28, 28)
# F.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
sobel_output = F.conv2d(sample_28x28, sobel_filter)

print(f"Sobel 필터 출력 크기: {sobel_output.shape}")