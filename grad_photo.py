from utils.setup_dataset import get_train_dataset
from utils.setup_model import get_model
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy
from utils.setup_dataset import get_train_dataset
from matplotlib.pyplot import cm


if __name__ == '__main__':
    PATH = './model_MNIST.pth'
    (train_loader, valid_loader), batch_size = get_train_dataset(trial=None)
    model, reduction_val = get_model(trial=None)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    model = model[0]
    model = model[:-1]
    model = model.eval()
    output_target = [0, 0, 0]
    for _, (data, labels) in enumerate(train_loader):
        break

    imgs = data[:1]
    imgs2 = data[1:2]
    plt.imsave(f'./saving_folder/input.png', imgs.clone().detach().squeeze(dim=0).squeeze(dim=0), cmap='gray')
    plt.imsave(f'./saving_folder/target.png', imgs2.clone().detach().squeeze(dim=0).squeeze(dim=0), cmap='gray')

    net = model.eval().cpu()

    # run the images through the model
    output = net(imgs2)
    # pred = output.detach().float() # torch.argmax(output, 1).float()
    pred = torch.tensor(output_target, dtype=torch.float, requires_grad=False).unsqueeze(0)
    # pred[0][2] = pred[0][2] + 10
    # pred = torch.unsqueeze(pred, dim=0)
    # prepare loss function for the method

    loss = nn.MSELoss(reduction='sum')

    # inputs need to have gradients enabled!
    imgs.requires_grad = True

    lr = 0.0001
    min_lr = 0.0001
    for i in range(100000):
    # run the model and calculate the loss
        outputs = net(imgs)
        cost = loss(outputs, pred)
        if i % 50 == 0:
            print(cost)
        if(cost <= 0.002):
            plt.imsave(f'./saving_folder/xxxx{i}_{cost.data}.png', imgs.clone().detach().squeeze(dim=0).squeeze(dim=0),
                       cmap='gray')
            break
    # get input gradients
        grad = torch.autograd.grad(cost, imgs, create_graph=False)[0]
        if i % 1000 == 0:
            plt.imsave(f'./saving_folder/xxxx{i}_{cost.data}.png', imgs.clone().detach().squeeze(dim=0).squeeze(dim=0),
                       cmap='gray')
        imgs = imgs - lr * grad

        if i % 1000 == 0:
            print("DECREASE LR")
            lr = max(min_lr, lr * 0.1)
