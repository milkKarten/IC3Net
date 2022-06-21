import numpy as np
import torch
from torch import optim
import torch.nn as nn
#
#
# arr = torch.zeros((4,3))
# arr[:,-2:] = 1
# print (arr)

model = nn.Sequential(
    nn.Linear(2, 2),
    nn.Sigmoid(),
    nn.Linear(2, 2)
)

print (model[0].weight.detach().clone())

optimizer = optim.SGD(model.parameters(), lr=1.0)
criterion = nn.CrossEntropyLoss()

batch_size = 10
x = torch.randn(batch_size, 2)
target = torch.randint(0, 2, (batch_size,))


# Create Gradient mask
gradient_mask = torch.zeros(2, 2)
gradient_mask[0, 0] = 1.0
model[0].weight.register_hook(lambda grad: grad.mul_(gradient_mask))

# Get weight before training
w0 = model[0].weight.detach().clone()

# Single training iteration
optimizer.zero_grad()
output = model(x)


loss = criterion(output, target)

loss.backward(retain_graph=True)
print('Gradient: ', model[0].weight.grad)

# Create Gradient mask
gradient_mask = torch.ones(2, 2)
gradient_mask[0, 0] = 0
model[0].weight.register_hook(lambda grad: grad.mul_(gradient_mask))
loss2 = 2*criterion(output, target)
loss2.backward()
print('Gradient: ', model[0].weight.grad)


optimizer.step()

# Compare weight update
w1 = model[0].weight.detach().clone()
print('Weights updated ', w0!=w1)
