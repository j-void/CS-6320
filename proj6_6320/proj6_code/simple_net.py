import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention
    to understand what it means
    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    ###########################################################################
    # Student code begin
    ###########################################################################

    self.cnn_layers = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),
      nn.MaxPool2d(3),
      nn.ReLU(),
      nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5),
      nn.MaxPool2d(3),
      nn.ReLU(),
      
    )
    
    self.fc_layers = nn.Sequential(
      nn.Linear(500, 100),
      nn.Linear(100, 15)
      
    )

    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')
    ###########################################################################
    # Student code end
    ###########################################################################

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    model_output = None
    ###########################################################################
    # Student code begin
    ###########################################################################

    x = self.cnn_layers(x)
    x = x.view(-1, 500)
    model_output = self.fc_layers(x)

    ###########################################################################
    # Student code end
    ###########################################################################
    return model_output

  # Alternative implementation using functional form of everything
class SimpleNet2(nn.Module):
  def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(1, 10, 5)
      self.pool = nn.MaxPool2d(3, 3)
      self.conv2 = nn.Conv2d(10, 20, 5)
      self.fc1 = nn.Linear(500, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, 15)
      self.loss_criterion = nn.CrossEntropyLoss()

  def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = x.view(-1, 500)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x
