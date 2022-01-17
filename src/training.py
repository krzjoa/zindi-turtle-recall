#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 22:06:11 2022

@author: krzysztof
"""
from turtle_net_1 import turtle_net_1
from turtle_dataset import TurtleDataset


# Loading data
TRAIN_CSV_PATH = "../data/train.csv"
TEST_CSV_PATH = "../data/test.csv"
EXTRA_IMAGES_PATH = "../data/extra_images.csv"
SAMPLE_SUBMISSION_PATH = "../data/sample_submission.csv"
IMAGES_PATH = "../data/images"

# CSV files
train_csv = pd.read_csv(TRAIN_CSV_PATH)
test_csv  = pd.read_csv(TEST_CSV_PATH)
extra_images = pd.read_csv(EXTRA_IMAGES_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

all_imgs = pd.concat([train_csv, extra_images, test_csv])


# List of images
imgs = os.listdir(IMAGES_PATH, )
imgs = [os.path.join(IMAGES_PATH, x) for x in imgs]




train_turtle_dl = torch_data.DataLoader(
    turtle_dataset, batch_size=32, shuffle=True, drop_last=True)


# Training loop

optimizer = optim.Adam(turtle_net_1.parameters())
loss_fun = nn.CrossEntropyLoss()

epochs = 5




for e in range(epochs):
    train_loss = 0.0
    for img, pos, turtle_id in tqdm(train_turtle_dl):
        
        # Clear the gradients
        optimizer.zero_grad()
        # Forward Pass
        target = turtle_net_1(img).squeeze()
        # Find the Loss
        # print(target.shape, turtle_id.shape)
        loss = loss_fun(target, turtle_id)
        # Calculate gradients 
        loss.backward()
        # Update Weights
        optimizer.step()
        # Calculate Loss
        train_loss += loss.item()
    
    print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_turtle_dl)}')