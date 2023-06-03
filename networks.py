import torch
import torch.nn as nn
import copy

def antiderivTanh(x): # activation function aka the antiderivative of tanh
    return torch.abs(x) + torch.log(1+torch.exp(-2.0*torch.abs(x)))


class ResNN(nn.Module):
    def __init__(self, din, m, dout, act = nn.Softplus(), nTh=2, tau_learn = True, h=1):
        """
        :param d:   int, dimension of space input (expect inputs to be d+1 for space-time)
        :param m:   int, hidden dimension
        :param nTh: int, number of resNet layers , (number of theta layers)
        """
        super().__init__()

        if nTh < 2:
            print("nTh must be an integer >= 2")
            exit(1)

        self.din = din
        self.dout = dout
        self.m = m
        self.nTh = nTh
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(din, m, bias=True)) # opening layer
        self.layers.append(nn.Linear(m,m, bias=True)) # resnet layers
        if tau_learn:
            self.taus = nn.Parameter(torch.rand(nTh-1, requires_grad=True))
        else:
            self.taus = (1.0 / (self.nTh-1))*torch.ones(nTh-1)
        for i in range(nTh-2):
            self.layers.append(copy.deepcopy(self.layers[1]))
        self.layers.append(nn.Linear(m, dout)) # output layer
        self.act = act

    def forward(self, x):
        """
        :param x: tensor nex-by-d+1, inputs
        :return:  tensor nex-by-1,   outputs
        """

        x = self.act(self.layers[0].forward(x))

        for i in range(1,self.nTh):
            x = x + self.taus[i-1] * self.act(self.layers[i](x))
            # x = x + self.h * self.act(self.layers[i](x))
        x = self.layers[-1](x)

        return x


class fDNN(nn.Module):
    def __init__(self, din, m, dout, gamm=torch.tensor(0.5), act=nn.Softplus(), nTh=3, tau_learn = True, h=1):
        """
        :param din:    int, dimension of space input
        :param dout:   int, dimension of space output
        :param m:      int, hidden dimension
        :param gamm:   fractional power
        :param nTh:    int, number of factional net layers , (number of theta layers)
        """
        super().__init__()

        if nTh < 3:
            print("nTh must be an integer >= 3")
            exit(1)

        self.din = din
        self.dout = dout
        self.m = m
        self.gamm = gamm
        self.nTh = nTh
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(din, m, bias=True))  # opening layer
        self.layers.append(nn.Linear(m, m, bias=True))  # resnet layers
        if tau_learn:
            self.taus = nn.Parameter(torch.rand(nTh-1, requires_grad=True))
        else:
            self.taus = (((1.0 / (self.nTh-1)) ** gamm) * torch.exp(torch.lgamma(2 - gamm)))*torch.ones(nTh-1)
        for i in range(nTh - 2):
            self.layers.append(copy.deepcopy(self.layers[1]))
        self.layers.append(nn.Linear(m, dout))  # output layer
        self.act = act

    def forward(self, x):
        """
        :param x: tensor nex-by-d+1, inputs
        :return:  tensor nex-by-1,   outputs
        """

        x = self.act(self.layers[0].forward(x))
        xall = x.unsqueeze(-1)
        for j in range(1, self.nTh):
            suma2 = 0.0 * x  # initialize sum for fractional derivative
            # print(f'{suma2.shape} at {j}')
            for k in range(j - 1):  # compute fractional deriavtives
                suma2 = suma2 + ((j + 1 - k) ** (1 - self.gamm) - (j - k) ** (1 - self.gamm)) * (
                            xall[:, :, k + 1] - xall[:, :, k])
            x = x + suma2 + (self.taus[j-1] ** self.gamm) * torch.exp(torch.lgamma(2 - self.gamm)) \
                * self.act(self.layers[j](x))
            xall = torch.cat((xall, x.unsqueeze(-1)), -1)
        x = self.layers[-1](x)

        return x


if __name__ == "__main__":
    din, m, dout = 4, 3, 1
    net = fDNN(din, m, dout, nTh=3)

    x0 = torch.rand(3,4)
    net(x0)
