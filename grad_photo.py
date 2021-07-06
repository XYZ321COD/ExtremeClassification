from utils.setup_dataset import get_dataset
from utils.setup_model import get_model
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy
from utils.setup_dataset import get_dataset
from matplotlib.pyplot import cm


if __name__ == '__main__':
    PATH = './model_MNIST.pth'
    (train_loader, valid_loader), batch_size = get_dataset(trial=None)
    model, reduction_val = get_model(trial=None)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    model = model[0]
    model = model[:-1]
    model = model.eval()
    output_target = [120, 120, 120]
    for _, (data, labels) in enumerate(train_loader):
        break

    imgs = data[:1]
    imgs2 = data[1:2]
    plt.imsave(f'./saving_folder/input.png', imgs.clone().detach().squeeze(dim=0).squeeze(dim=0))
    plt.imsave(f'./saving_folder/target.png', imgs2.clone().detach().squeeze(dim=0).squeeze(dim=0))

    net = model.eval().cpu()

    # run the images through the model
    output = net(imgs)
    pred = output
    pred[0][2] = pred[0][2] + 2
    # pred = torch.tensor(pred, dtype=torch.float, requires_grad=True)
    # pred = torch.unsqueeze(pred, dim=0)
    # prepare loss function for the method
    loss = nn.MSELoss(reduction='sum')

    # eps parameter for the FGSM
    eps = 0.02
    # inputs need to have gradients enabled!
    imgs.requires_grad = True
    for i in range(100000):
    # run the model and calculate the loss
        outputs = net(imgs)
        cost = loss(outputs, pred)
        if(cost <= 0.02):
            break
    # get input gradients
        grad = torch.autograd.grad(cost, imgs, create_graph=False)[0]
        if i % 10000 == 0:
            plt.imsave(f'./saving_folder/xxxx{i}_{cost.data}.png', imgs.clone().detach().squeeze(dim=0).squeeze(dim=0))
        imgs = imgs - 0.0001 * grad
    print(cost)   


