import torch

class Softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        exp_x = torch.exp(x)
        return exp_x / torch.sum(exp_x)

class SoftmaxStable(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        c = torch.max(x)
        exp_x = torch.exp(x - c)
        return exp_x / torch.sum(exp_x)

class SigmoidStable(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1 / (1 + torch.exp(-x))
    


# Example
data = torch.tensor([1, 2, 3])
softmax = Softmax()
softmax_stable = SoftmaxStable()
sigmoid_stable = SigmoidStable()
print("Sigmoid Stable:", sigmoid_stable(data))
print("Softmax:", softmax(data))
print("Stable Softmax:", softmax_stable(data))
