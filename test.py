import torch
import torchvision
import torchaudio

print(torch.__version__)
print(torchvision.__version__)
print(torchaudio.__version__)

print(torch.cuda.is_available())  # True가 출력되어야 CUDA가 설치되어 있음
print(torch.version.cuda)          # 설치된 CUDA 버전 확인