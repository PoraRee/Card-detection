import cv2
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import os
import numpy as np

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((40,40)),
    T.ToTensor(),
    T.Normalize(0.5, 0.5),
])

SHAPE = 1600
HIDDEN = 100
WEIGHT_PATH = "best_model.pt"
IDX2CLASS = ['7C', '5D', '4D', 'AD', '2D', '2H', '10H', '8H', '3S', '6D', '8C', '10C', '8D', '6C', '3D', '6H', 'AH', 'AS', '5H', 'JC', 'JS', '3H', '10S', 'JH', 'JD', '4S', '3C', '9H', '4H', 'AC', '5S', '9C', '9S', 'QD', '7H', 'KC', '4C', '10D', '8S', '6S', '7D', 'KS', '7S', '5C', '9D', 'QC', '2C', 'KD', 'QH', 'QS', '2S', 'KH']
CLASS2IDX = {k:v for v, k in enumerate(IDX2CLASS)}


class CardNet(nn.Module):
    def __init__(self):
        super(CardNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(SHAPE, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, 52)

    def forward(self, input):
        x = self.pool1(F.relu(self.conv1(input)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(-1, SHAPE)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Classifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CardNet().to(self.device)
        self.model.load_state_dict(torch.load(WEIGHT_PATH, map_location=self.device))
        self.transform = transform
        self.mapper = IDX2CLASS

    def get_class(self, input):
        # input = input[:input.shape[0]//2, :input.shape[1]//2]
        x = transform(input)
        x = x.to(self.device)
        pred =  self.model(x)
        result = pred.argmax().item()
        class_result = self.mapper[result]

        return class_result


def p(img):
    x = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    _, x = cv.threshold(x, 200, 255, cv.THRESH_BINARY)
    return x

class Classifier2():
    TEMPLATE_PATH = "../bicycleimg_cleaned/"
    
    def __init__(self):
        cache = []
        self.label = []
        for fn in os.listdir(self.TEMPLATE_PATH):
            timg = cv.imread(os.path.join(self.TEMPLATE_PATH, fn))
            timg = cv.cvtColor(timg, cv.COLOR_BGR2RGB)
            timg = p(timg)
            cache.append(timg)
            self.label.append(fn.split('-')[0])
        self.templates = np.stack(cache)


    def get_class(self, inp):
        out = p(inp)

        scores = np.mean(np.power(np.expand_dims(out, 0) - self.templates, 2).reshape(self.templates.shape[0], -1), axis=1)
        a_score = np.argmin(scores)

        return self.label[a_score]


if __name__ == "__main__":
    
    classifer = Classifier()
    img = cv2.imread("../cards/2_of_clubs.png")[:,:,::-1]

    card_class = classifer.get_class(img)
    print(card_class)
