import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(object):

    def __init__(self, device, ngpu):

        self.device, self.ngpu = device, ngpu
        self.model = SRNET(self.ngpu).to(self.device)
        if (self.device.type == 'cuda') and (self.model.ngpu > 0):
            self.model = nn.DataParallel(self.model, list(range(self.model.ngpu)))

        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        print(self.model)
        print("The number of parameters: {}".format(num_params))

        self.mse = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3)

class SRNET(nn.Module):

    def __init__(self, ngpu):
        super(SRNET, self).__init__()

        self.ngpu = ngpu
        self.model = nn.Sequential(
          nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=9//2),
          nn.ReLU(),
          nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0//2),
          nn.ReLU(),
          nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, stride=1, padding=5//2),
          nn.ReLU(),
        )

    def forward(self, input):
        return torch.clamp(self.model(input), min=1e-12, max=1-(1e-12))
