import torch.nn as nn
import torch


class classification_net(nn.Module):

    def __init__(self, negative_slope=0.2, dropout_percentage=0.3):
        super(classification_net, self).__init__()

        self.dropout = nn.Dropout(dropout_percentage, inplace=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(14 * 14 * 64, 512),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
#         self.fc2 = nn.Sequential(
#             nn.Linear(1024, 512),
#             nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
#         )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 101)
        )

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(input_image)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        fc1 = self.fc1(torch.flatten(conv4, 1))
#         fc2 = self.fc2(fc1)
        fc3 = self.fc3(fc1)
        return fc3

