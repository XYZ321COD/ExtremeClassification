import matplotlib.pyplot as plt
import torch
import yaml
from utils.setup_dataset import get_datasets
from utils.setup_model import get_model
import sklearn.metrics as mt
import numpy




def ood_auroc(in_distr_probs, out_distr_probs):
     in_distr_confidences, _ = in_distr_probs.max(dim=-1)
     out_distr_confidences, _ = out_distr_probs.max(dim=-1)
     confidences = torch.cat([in_distr_confidences,
out_distr_confidences]).cpu().numpy()
     ood_labels = torch.cat([torch.ones_like(in_distr_confidences),
torch.zeros_like(out_distr_confidences)]).cpu().numpy()
     auroc = mt.roc_auc_score(ood_labels, confidences)
     return auroc
 
 
if __name__ == '__main__':
    auroc_array = []
    for u in range(5):
        file = open(r'config.yaml')
        cfg = yaml.load(file, Loader=yaml.FullLoader)
        file_visual = open('config_visual.yaml')
        cfg_vis = yaml.load(file_visual, Loader=yaml.FullLoader)

        DEVICE = torch.device(cfg['options']['device'])
        PATH = cfg_vis['model_path']+str(u)
        
        (train_train_loader, train_valid_loader), (eval_train_loader, eval_valid_loader) = get_datasets(trial=None, name='CIFAR10-CIFAR100')

        model, reduction_val = get_model(trial=None)
        model.load_state_dict(torch.load(PATH))
        model.eval()
        model = model.to(DEVICE)
        auroc_final = 0    
        with torch.no_grad():
            for (data, target), (data_un, target_un) in zip(train_valid_loader, eval_valid_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                data_un, target_un = data_un.to(DEVICE), target_un.to(DEVICE)
                in_distr_probs = model(data)
                out_distr_probs = model(data_un)
                auroc_final += (ood_auroc(in_distr_probs, out_distr_probs))
            auroc_final = auroc_final / len(train_valid_loader)
            auroc_array.append(auroc_final)
        print(auroc_array)
        print(numpy.std(auroc_array))
        print(numpy.mean(auroc_array))

