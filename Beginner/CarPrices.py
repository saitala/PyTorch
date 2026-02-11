import pandas as pd
from pathlib import Path
import os
import torch
import torch.nn

currentpath = os.getcwd()
df = pd.read_csv(currentpath+'/Beginner/data/car_prices.csv')
input_attributes = df[['year', 'mileage']].values.astype(float)
output_attributes = df[['price']].values.astype(float)

max_input_attributes = input_attributes.max(axis=0)
min_input_attributes = input_attributes.min(axis=0)

max_output_attributes = output_attributes.max(axis=0)
min_output_attributes = output_attributes.min(axis=0)

normalized_input_attributes = (input_attributes - min_input_attributes)/ (max_input_attributes - min_input_attributes)
normalized_output_attributes = (output_attributes - min_output_attributes) / (max_output_attributes - min_output_attributes)

input_tensor = torch.tensor(normalized_input_attributes, dtype=torch.float32)
output_tensor = torch.tensor(normalized_output_attributes, dtype=torch.float32)

model = torch.nn.Linear(2,1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

epoch=2000
for epoch in range(epoch):
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, output_tensor)
    loss.backward()
    optimizer.step()

    # Optional: print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epoch}], Loss: {loss.item():.4f}')



