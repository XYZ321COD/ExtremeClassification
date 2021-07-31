from utils.setup_dataset import get_dataset
from utils.setup_model import get_model
from utils.loss_functions.loss_functions import str_to_loss
import torch.nn as nn
import torch.optim as optim
import torch
import yaml
import sklearn.metrics as mt
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib
file = open(r'config.yaml')
cfg = yaml.load(file, Loader=yaml.FullLoader)

def objective(trial, name_of_the_run=cfg['options']['name_of_the_run']):
    
    DEVICE = torch.device(cfg['options']['device'])
    EPOCHS = cfg['options']['epochs']

    model, reduction_val = get_model(trial=trial)
    model.to(DEVICE)

    optimizer_name = trial.suggest_categorical("optimizer", cfg['hyperparameters']['optimizers'])
    lr = trial.suggest_float("lr", min(cfg['hyperparameters']['lr']), max(cfg['hyperparameters']['lr']))
    WRITTER = SummaryWriter('{}/{}_{}_red_{}'.format(name_of_the_run, optimizer_name, lr, reduction_val))

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    criterion = str_to_loss(cfg['options']['loss_function'])
    (train_loader, valid_loader), batch_size = get_dataset(trial=trial)
    
    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for (data, target) in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            if cfg['options']['loss_function'] == "BCELoss":
                target = nn.functional.one_hot(target, num_classes=cfg['dataset']['num_classes']).to(dtype=torch.float32)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        # Validation of the model.
        model.eval()
        accuracy = 0
        loss = 0
        with torch.no_grad():
            for (data, target) in valid_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                loss += criterion(output, nn.functional.one_hot(target, num_classes=cfg['dataset']['num_classes']).to(dtype=torch.float32)  if cfg['options']['loss_function'] == "BCELoss"
                                  else target)
                accuracy += mt.accuracy_score(target.cpu().detach(), output.cpu().detach().argmax(dim=1))        
                
        accuracy_full = accuracy / len(valid_loader)
        loss_full = loss / len(valid_loader)
        
        for n, p in model.named_parameters():
            if 'bias' not in n:
                WRITTER.add_histogram('{}'.format(n), p, epoch)
                if p.requires_grad:
                    WRITTER.add_histogram('{}.grad'.format(n), p.grad, epoch)
        WRITTER.add_scalar('BCE Loss',(loss_full.item()), epoch+1)
        WRITTER.add_scalar('Acc', accuracy_full, epoch+1)
        
        trial.report(accuracy_full, epoch+1)
        trial.report(loss_full.detach(), epoch+1)
        print(f'Accuracy {accuracy_full} in epoch {epoch+1}')
    trial.set_user_attr("model", cfg['options']['model'])
    trial.set_user_attr("dataset", cfg['dataset']['name'])
    trial.set_user_attr('bce_loss', loss_full.item())
    trial.set_user_attr('epochs', EPOCHS)  
    if cfg['options']['save_model']:
        torch.save(model.state_dict(), f'./model_{cfg["dataset"]["name"]}_{cfg["options"]["loss_function"]}_bias{cfg["options"]["bias"]}_{round(accuracy_full, 3)}')
    
    return accuracy_full
  
        # Handle pruning based on the intermediate value.
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()
