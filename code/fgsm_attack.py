# header files needed
import numpy as np
import torch
import torch.nn as nn
import torchvision


# FGSM attack code(reference: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


# constants
model_path = "/home/arpitdec5/Desktop/adversarial_attacks_neural_networks/code/model_vgg16_cifar10.pth"
files = "/home/arpitdec5/Desktop/adversarial_attacks_neural_networks/data/"
classes_map = {"0": "airplane", "1": "automobile", "2": "bird", "3": "cat", "4": "deer", "5": "dog", "6": "frog", "7": "horse", "8": "ship", "9": "truck"}
actual_target = torch.tensor([3])

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
    torch.nn.Linear(512, 10),
    torch.nn.Softmax(dim=1)
)
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

# predict output for different epsilons
model.eval()
epsilons = [0, .05, .1, .15, .2, .25, .3]
for epsilon in epsilons:
    for i, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)
        input.requires_grad = True
        output = model(input)
        _, actual_pred = output.max(1)

        # loss
        loss = torch.nn.functional.nll_loss(output, actual_target)
        
        # backward pass
        model.zero_grad()
        loss.backward()
        grad = input.grad.data

        # fgsm attack
        fgsm_input = fgsm_attack(input, epsilon, grad)
        output = model(fgsm_input)

        # check fgsm output with actual output
        fgsm_pred = output.max(1, keepdim=True)[1]
        if(fgsm_pred.item() != actual_target.item()):
            print("Adversarial example!")
            #print("Adversarial example: " + classes_map[str(int(fgsm_pred.item()))])
