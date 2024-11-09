import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

model = torch.nn.Linear(2,4)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

for epoch in range(100):
    scheduler.step()
    print(optimizer.param_groups[0]['lr'])