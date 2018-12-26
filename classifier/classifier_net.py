import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self,in_features=2048, num_class = 145):
        super(Classifier,self).__init__()
        self.num_class = 145
        self.fc = nn.Linear(in_features=in_features,out_features=145)
        self.fc2 = nn.Linear(in_features=in_features,out_features=2)

    def forward(self,x):
        input = x
        x = self.fc(input)
        y = self.fc2(input)
        return (x,y)



if __name__ =='__main__':
    classifier = Classifier(2048,145)

    input = torch.Tensor(2048)

    out  = classifier(input)

    print(out)