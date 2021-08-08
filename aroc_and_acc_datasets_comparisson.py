import matplotlib.pyplot as plt
import torch
import yaml
from utils.setup_dataset import get_datasets
from utils.setup_model import get_model
import sklearn.metrics as mt
import numpy
from scipy.stats import pearsonr
import collections



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
    acc_array = []
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
        accuracy = 0
        with torch.no_grad():
            for (data, target), (data_un, target_un) in zip(train_valid_loader, eval_valid_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                data_un, target_un = data_un.to(DEVICE), target_un.to(DEVICE)
                in_distr_probs = model(data)
                out_distr_probs = model(data_un)
                auroc_final += (ood_auroc(in_distr_probs, out_distr_probs))
                accuracy += mt.accuracy_score(target.cpu().detach(), in_distr_probs.cpu().detach().argmax(dim=1))        
                    
            accuracy_full = accuracy / len(train_valid_loader)
            auroc_final = auroc_final / len(train_valid_loader)
            auroc_array.append(auroc_final)
            acc_array.append(accuracy_full)
    # print(f' Accuracy {acc_array}')
    # print(f' AUROC {auroc_array}')
    mapping = collections.OrderedDict(sorted(dict(zip(acc_array, auroc_array)).items()))
    # print(mapping.keys())
    # print(mapping.values())
    print(f' Accuracy {mapping.keys()}')
    print(f' AUROC {mapping.values()}')
    print(f' std auroc {numpy.std(auroc_array)}')
    print(f' mean auroc {numpy.mean(auroc_array)}')
    print(f' std acc {numpy.std(acc_array)}')
    print(f' mean acc {numpy.mean(acc_array)}')
    print(f' pearsonr {pearsonr(acc_array, auroc_array)}')

