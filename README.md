# Adversarial Attacks on Neural Networks

[![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)](LICENSE.md)
---


### Author
Arpit Aggarwal


### Introduction to the Project 
This project surveys the various adversarial attacks on neural networks. This is useful in cases where we deploy the deep learning systems in real-world scenarios like autonomous driving. First, different CNN architectures(VGG-16, VGG-19 and ResNet-50) are trained on CIFAR-10 dataset. Then after training these architectures, various adversarial examples from CIFAR-10 dataset are generated and consequently these examples are used to evaluate how robust these trained models are to various forms of adversarial attacks. 


### Data
The data used for this task was CIFAR-10 dataset. The dataset has been divided into two sets: Training data and Validation data. The analysis of different CNN architectures for image classifcation on CIFAR-10 dataset was done on comparing the Training Accuracy and Validation Accuracy values.


### Results
The results after using different CNN architectures on CIFAR-10 dataset are given below:

1. <b>ResNet-50(pretrained on ImageNet dataset)</b><br>

Training Accuracy = 97.31% and Validation Accuracy = 82.63% (e = 100, lr = 0.001, m = 0.9, bs = 64, wd = 5e-4)<br>


2. <b>VGG-16(pretrained on ImageNet dataset)</b><br>

Training Accuracy = 97.55% and Validation Accuracy = 87.63% (e = 100, lr = 0.001, m = 0.9, bs = 64, wd = 5e-4)<br>


3. <b>VGG-19(pretrained on ImageNet dataset)</b><br>

Training Accuracy = 98.78% and Validation Accuracy = 88.35% (e = 100, lr = 0.001, m = 0.9, bs = 64, wd = 5e-4)<br>


### Software Required
To run the jupyter notebooks, use Python 3. Standard libraries like Numpy and PyTorch are used.


### Credits
The following links were helpful for this project:
1. https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
2. https://www.youtube.com/channel/UC88RC_4egFjV9jfjBHwDuvg
3. https://github.com/pytorch/tutorials
