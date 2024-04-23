#!/usr/bin/env python3
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

data = np.genfromtxt('/home/pplonka/Pobrane/Cefeidy.txt', dtype=[('Nazwa', 'S20'), ('Typ', 'S5'), ('P1', 'f8'), ('A1', 'f8'), ('R2_1', 'f8'), ('phi21_1', 'f8'), ('R31_1', 'f8'), ('phi31_1', 'f8')])
for field in data.dtype.names:
    data[field] = np.where(data[field] == -99.99, 0, data[field])

C1 = data['P1']
C2 = data['R2_1']
C3 = data['phi21_1']
C4 = data['R31_1']
C5 = data['phi31_1']
C6 = data['A1']
poddtyp = data['Typ']
Tensor_1 = np.stack((C1,C2,C3,C4,C5,C6), axis=1)


unique_values = np.unique(poddtyp)
class_to_idx = {val: idx for idx, val in enumerate(unique_values)}
class_indices = np.array([class_to_idx[val] for val in poddtyp])

x_train, x_test, y_train, y_test = train_test_split(Tensor_1,
                                                    class_indices,
                                                    test_size=0.2)

x_train = torch.from_numpy(x_train).float().to(device)
y_train = torch.from_numpy(y_train).long().to(device)
x_test = torch.from_numpy(x_test).float().to(device)
y_test = torch.from_numpy(y_test).long().to(device)

class OGLE(nn.Module):
    def __init__(self):
        super(OGLE, self).__init__()
        self.l1 = nn.Linear(6, 20000)
        self.l1_1 = nn.Linear(20000, 10000)
        self.l2 = nn.Tanh()
        self.l3 = nn.Linear(10000, len(unique_values))  

    def forward(self, x):
        x = self.l2(self.l1_1(self.l2(self.l1(x))))
        return self.l3(x)
    
model = OGLE().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

epochs = 300

scaler = GradScaler()
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    with autocast():
        y_logits = model(x_train)
        loss = loss_fn(y_logits, y_train)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    if epoch % 10 == 0:
        with torch.no_grad():
            model.eval()
            a = model(x_test)
            y_pred = torch.argmax(a, dim=1)
            acc = torch.sum(y_pred == y_test).item() / len(y_test) * 100
            val_loss = loss_fn(a, y_test)
            print(f"Epoch {epoch}: Test Accuracy: {acc:.2f}% Loss: {loss}")
