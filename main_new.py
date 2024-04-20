import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import collections
from model_definition import LightWeightEEG2ClassificationCNN

if not hasattr(collections, "Callable"):
    collections.Callable = collections.abc.Callable


# 데이터 로드 함수
def load_data(base_path="./data/", num_subjects=9):
    data_X, data_y = [], []
    for i in range(1, num_subjects + 1):
        X = np.load(f"{base_path}S0{i}_train_X.npy")
        y = np.load(f"{base_path}S0{i}_train_y.npy")
        data_X.append(X)
        data_y.append(y)
    data_X = np.concatenate(data_X, axis=0)
    data_y = np.concatenate(data_y, axis=0)
    print(data_X.shape)
    print(data_y.shape)
    return data_X, data_y


# 사용자 정의 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(
            self.y[idx], dtype=torch.long
        )


# 학습 함수
import pdb


def train_model(dataloader, model, criterion, optimizer, num_epochs=5):
    model.to(device)
    criterion.to(device)
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 데이터 로드 및 DataLoader 설정
data_X, data_y = load_data()  # 데이터 로드
dataset = CustomDataset(data_X, data_y)  # 데이터셋 객체 생성
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)  # DataLoader 생성

# 모델 인스턴스 생성 (주석 처리)
model = LightWeightEEG2ClassificationCNN(input_size=1125, output_size=4)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
train_model(dataloader, model, criterion, optimizer)  # 학습 실행
