from classifier_net import Classifier
from dataset import ClassifierDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import numpy as np

def main():

    file_root = "F:\\final_faces_feature_resnet"
    train_txt = "F:\\images\\train_with_normal.txt"
    val_txt = "F:\\images\\val_with_normal.txt"

    #0. hyper parameters
    lr = 0.01
    momentum = 0.9
    batch_size = 128
    epoch = 25
    #1. model
    model = Classifier(in_features=2048,num_class=145)

    #2. dataset 
    train_data = ClassifierDataset(file_root=file_root,train_txt=train_txt)
    val_data = ClassifierDataset(file_root=file_root,train_txt = val_txt)

    train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data,batch_size=batch_size,shuffle=False,num_workers=4)

    #3. loss function
    criterion = nn.CrossEntropyLoss()

    #4. optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    #5. scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=3)
    
    #6. train
    train_loss = []
    val_loss = []
    best_score = 0
    
    for e in range(epoch):
        right = 0
        total = 0
        print(F'start epoch====={e}')
        for p in optimizer.param_groups:
            print(F'lr---{p["lr"]}')
        train_loss.clear()
        val_loss.clear()
        model.train()
        for i,data in enumerate(train_loader):
            label = data[1].type(torch.LongTensor)
            label2 = data[2].type(torch.LongTensor)
            out = model(data[0])
            # print(out.type(),label.type())
            loss = criterion(out[0],label)
            loss2 = criterion(out[1],label2)
            loss = loss+loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if i % 50 == 0:
                print(F'Batch::{i}, Loss::{loss.item()}')
        
        print(F'Train Epoch::{e}, Loss::{np.mean(train_loss)}')
        model.eval()
        for i,data in enumerate(val_loader):
            label = data[1].type(torch.LongTensor)
            # label2 = data[2].type(torch.LongTensor)
            out = model(data[0])[0]
            pred = F.softmax(out,dim=1)
            pred = torch.max(pred,dim=1)[1]
            correct = torch.sum(pred==label)
            right += correct
            total += len(label)
            loss = criterion(out,label)
            val_loss.append(loss.item())
        val_epoch_loss = np.mean(val_loss)
        acc = float(right)/total
        print(F'Val Epoch::{e}, Loss::{val_epoch_loss}, best-score: {best_score}, acc:: {acc}')
        scheduler.step(np.mean(val_loss))

        #save model
        if acc > best_score:
            best_score = acc
            # with open('classifier-best.pth','wb') as f:
            #     torch.save(model,f)
            torch.save(model.state_dict(),"classifier-best.pth")


if __name__=="__main__":
    main()