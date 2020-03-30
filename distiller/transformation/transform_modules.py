import torch.nn as nn

__all__ = ['PathThrough']

class PathThrough(nn.Module):
    def __init__(self):
        super(PathThrough, self).__init__()

    def forward(self, input):
        return input

    def __repr__(self):
        return self.__class__.__name__ + ' : only path through input to next layer'

class PathThrough_test0(nn.Module):
    def __init__(self):
        super(PathThrough_0, self).__init__()

    def forward(self, input):
        return input

    def __repr__(self):
        return self.__class__.__name__ + ' : only path through input to next layer'

class PathThrough_test1(nn.Module):
    def __init__(self):
        super(PathThrough_1, self).__init__()

    def forward(self, input):
        return input

    def __repr__(self):
        return self.__class__.__name__ + ' : only path through input to next layer'