import matplotlib.pyplot as plt
import matplotlib
import torch
import yaml
from utils.setup_dataset import get_dataset
from utils.setup_model import get_model
if __name__ == '__main__':
    
    file = open(r'config.yaml')
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    file_visual = open('config_visual.yaml')
    cfg_vis = yaml.load(file_visual, Loader=yaml.FullLoader)

    DEVICE = torch.device(cfg['options']['device'])
    PATH = cfg_vis['model_path']
    
    (train_loader, valid_loader), batch_size = get_dataset(trial=None)

    model, reduction_val = get_model(trial=None)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    model = model.to(DEVICE)
    
    colors = ['red', 'green','blue','purple', 'black', 'cyan', 'brown', 'orange','deeppink','gray']
    
    
    x = []
    y = []
    z = []
    targets = []
    model = model[0]
    if cfg['options']['loss_function'] == 'BCELoss':
        model = model[:-1]
    print(model)
    with torch.no_grad():
        for (data, target) in valid_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            for elem_out, elem_targ in zip(output, target):
                x.append(elem_out[0].cpu())
                y.append(elem_out[1].cpu())
                z.append(elem_out[2].cpu())
                targets.append(elem_targ.cpu())
    
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection ="3d")
    sctt = ax.scatter3D(x, y, z, c=targets, cmap=matplotlib.colors.ListedColormap(colors))
    ax.legend()
    fig.colorbar(sctt, ax = ax)

    # plt.show()
    plt.savefig(f'fig_{PATH}.svg', format="svg")