import model
import torch
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# load dataset
data = pd.read_csv('data.csv', sep=';')

# split dataset as train-set and test-set
train, test = train_test_split(data, test_size=0.4)

# create train and test batch by DataLoader
train_data = DataLoader(ChallengeDataset(train.reset_index(drop=True), 'train'), batch_size=16, shuffle=True)
val_test_data = DataLoader(ChallengeDataset(test.reset_index(drop=True), 'val'), batch_size=16, shuffle=True)

# define train model
models = model.ResNet()

# calculate loss by nn.BCEloss
loss = torch.nn.BCELoss()

# optimizer by Adam method
optim = torch.optim.Adam(models.parameters(), lr=0.002, betas=(0.9, 0.999))

# train and fit
trainer = Trainer(model=models, crit=loss, optim=optim, train_dl=train_data, val_test_dl=val_test_data,
                  early_stopping_patience=10)
result = trainer.fit(100)

# show train process and result
plt.plot(np.arange(len(result[0])), result[0], label='train loss')
plt.plot(np.arange(len(result[1])), result[1], label='val loss')
plt.plot(np.arange(len(result[2])), result[2], label='F1 score')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
plt.show()
