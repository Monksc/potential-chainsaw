#!/usr/bin/env python3

import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
model.eval()


if __name__ == '__main__':
    import functions as f
    import numpy as np

    image = np.zeros((3, 7, 7))
    image = f.find_input(image, 64, 64, (0,1), np.eye(1000)[10], f.predict_model(model))
    f.to_image(image)
