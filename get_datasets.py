import torch
from torch import Tensor

def peaks(y: Tensor) -> Tensor:
    r"""
    Generate data from the `MATLAB 2D peaks function`_
    .. _MATLAB 2D peaks function: https://www.mathworks.com/help/matlab/ref/peaks.html
    :param y: (x, y) coordinates with shape :math:`(n_s, 2)` where :math:`n_s` is the number of samples
    :type y: torch.Tensor
    :return:
            - **f** (*torch.Tensor*) - value of peaks function at each coordinate with shape :math:`(n_s, 1)`
    """
    # function
    e1 = torch.exp(-1.0 * (y[:, 0]**2 + (y[:, 1] + 1)**2))
    f1 = 3 * (1 - y[:, 0])**2 * e1

    e2 = torch.exp(-y[:, 0]**2 - y[:, 1]**2)
    f2 = -10 * (y[:, 0] / 5 - y[:, 0]**3 - y[:, 1]**5) * e2

    e3 = torch.exp(-(y[:, 0] + 1)**2 - y[:, 1]**2)
    f3 = -(1 / 3) * e3
    f = f1 + f2 + f3
    return f.view(-1,1)

def sinosoindal(y: Tensor) -> Tensor:
    r"""
    Generate data for the sinusoidal`
    :param y: x coordinates with shape :math:`(n_s, 1)` where :math:`n_s` is the number of samples
    :type y: torch.Tensor
    :return:
            - **f** (*torch.Tensor*) - value of sinusoidal function at each coordinate with shape :math:`(n_s, 1)`
    """
    # function
    f = torch.sin(y)
    return f.view(-1,1)

if __name__=="__main__":
    import matplotlib.pyplot as plt
    # visualize peaks
    xy = torch.arange(-3, 3, 0.010, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(xy, xy, indexing='xy')
    grid_xy = torch.concat((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)), dim=1)
    grid_z = peaks(grid_xy)
    plt.imshow(grid_z.view(grid_x.shape))
    plt.show()