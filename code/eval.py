# header files needed
import numpy as np
import torch
import torch.nn as nn
import torchvision


# constants
model_path = "/home/arpitdec5/Desktop/adversarial_attacks_neural_networks/code/model_vgg16_cifar10.pth"
files = "/home/arpitdec5/Desktop/adversarial_attacks_neural_networks/data/"
classes_map = {"0": "airplane", "1": "automobile", "2": "bird", "3": "cat", "4": "deer", "5": "dog", "6": "frog", "7": "horse", "8": "ship", "9": "truck"}

# transforms
transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# dataset and loader
test_data = torchvision.datasets.ImageFolder(files, transform=transforms)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.vgg16_bn(pretrained=True)
model.avgpool = torch.nn.AvgPool2d((1, 1))
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(inplace=True),
    torch.nn.Dropout(),
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(inplace=True),
    torch.nn.Dropout(),
    torch.nn.Linear(512, 10)
)
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

# predict output
model.eval()
with torch.no_grad():
    for i, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)
        output = model(input)
        _, predicted = output.max(1)
        print("Prediction: " + classes_map[str(int(predicted[0]))])
