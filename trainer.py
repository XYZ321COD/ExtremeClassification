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
        WRITTER = SummaryWriter('{}/{}_{}_red_{}_classes_{}'.format(name_of_the_run, optimizer_name, lr, reduction_val, cfg['dataset'][['num_classes'][0]] ))

        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        criterion = str_to_loss(cfg['options']['loss_function'])
        (train_loader, valid_loader), batch_size = get_dataset(trial=trial)
        # Training of the model.
        for epoch in range(EPOCHS):
            loss_train = 0
            train_accuracy = 0
            model.train()
            for (data, target) in train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                if cfg['options']['loss_function'] == "BCELoss":
                    target = nn.functional.one_hot(target, num_classes=cfg['dataset']['num_classes']).to(dtype=torch.float32)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                loss_train += loss
                optimizer.step()
                train_accuracy += mt.accuracy_score(target.cpu().detach(), output.cpu().detach().argmax(dim=1))        

            # Validation of the model
            loss_full_train = loss_train / len(train_loader)
            train_accuracy_full = train_accuracy / len(train_loader)
            
            model.eval()
            accuracy_valid = 0
            loss_valid = 0
            with torch.no_grad():
                for (data, target) in valid_loader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    output = model(data)
                    loss_valid += criterion(output, nn.functional.one_hot(target, num_classes=cfg['dataset']['num_classes']).to(dtype=torch.float32)  if cfg['options']['loss_function'] == "BCELoss"
                                    else target)
                    accuracy_valid += mt.accuracy_score(target.cpu().detach(), output.cpu().detach().argmax(dim=1))        
                    
            accuracy_full_valid = accuracy_valid / len(valid_loader)
            loss_full_valid = loss_valid / len(valid_loader)
            
            for n, p in model.named_parameters():
                if 'bias' not in n:
                    WRITTER.add_histogram('{}'.format(n), p, epoch)
                    if p.requires_grad:
                        WRITTER.add_histogram('{}.grad'.format(n), p.grad, epoch)
            WRITTER.add_scalar('Loss_Train',(loss_full_train.item()), epoch+1)
            WRITTER.add_scalar('Acc_Train', train_accuracy_full, epoch+1)
            WRITTER.add_scalar('Acc_Valid', accuracy_full_valid, epoch+1)
            WRITTER.add_scalar('Loss_Valid',(loss_full_valid.item()), epoch+1)
            
            trial.report(accuracy_full_valid, epoch+1)
            trial.report(loss_full_valid.detach(), epoch+1)
            print(f'Accuracy {accuracy_full_valid} in epoch {epoch+1}')
        trial.set_user_attr("model", cfg['options']['model'])
        trial.set_user_attr("dataset", cfg['dataset']['name'])
        trial.set_user_attr('Loss_Valid', loss_full_valid.item())
        trial.set_user_attr('epochs', EPOCHS)  
        if cfg['options']['save_model']:
            torch.save(model.state_dict(), f'./model_{cfg["dataset"]["name"]}_{cfg["dataset"]["num_classes"]}{cfg["options"]["loss_function"]}_bias{cfg["options"]["bias"]}_{round(accuracy_full_valid, 3)}')
        
        plt.plot()
        return accuracy_full_valid
  
        # Handle pruning based on the intermediate value.
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

