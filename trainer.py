from utils.setup_dataset import get_dataset
from utils.setup_model import get_model
import torch.nn as nn
import torch.optim as optim
import torch
import yaml
import sklearn.metrics as mt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from visual import visualization
from utils.freeze_layer import *

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

    (train_loader, valid_loader), batch_size = get_dataset(trial=trial)
    
    # Get the W to visualize
    W = list(model.children())[-1].A
    visualization(name_of_the_run, optimizer_name, lr, reduction_val, 0, W)
    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()

        if epoch % 4 in (2, 3):
            model, W = freeze_all_except_first(model, DEVICE)
        elif epoch % 4 in (0, 1):
            model, W = freeze_first(model, DEVICE)

            # optimizer = getattr(optim, optimizer_name)(filter(lambda pr: pr.requires_grad, model.parameters()), lr=lr)

        for (data, target) in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            target = nn.functional.one_hot(target, num_classes=cfg['dataset']['number_of_classes']).to(dtype=torch.float32)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.BCELoss(reduction='sum')(output, target)
            loss.backward()
            optimizer.step()
        # Validation of the model.
        model.eval()
        accuracy = 0
        f1_score = 0
        loss = 0
        with torch.no_grad():
            for (data, target) in valid_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                target = nn.functional.one_hot(target, num_classes=cfg['dataset']['number_of_classes']).to(dtype=torch.float32)
                loss += nn.BCELoss(reduction='sum')(output, target)

                preds = torch.nn.functional.one_hot(output.argmax(dim=-1).cpu().detach(), num_classes=cfg['dataset']['number_of_classes'])
                accuracy += mt.accuracy_score(target.cpu().detach(), preds)
                f1_score += mt.f1_score(target.cpu().detach(), preds, average="samples")
        
        accuracy_full = accuracy / len(valid_loader)
        f1_score_full = f1_score / len(valid_loader)
        loss_full = loss / len(valid_loader)
        for n, p in model.named_parameters():
            if 'bias' not in n:
                WRITTER.add_histogram('{}'.format(n), p, epoch)
                if p.requires_grad:
                    WRITTER.add_histogram('{}.grad'.format(n), p.grad, epoch)
        WRITTER.add_scalar('BCE Loss',(loss_full.item()), epoch+1)
        WRITTER.add_scalar('Acc', accuracy_full, epoch+1)
        trial.report(accuracy_full, epoch+1)
        trial.report(f1_score_full, epoch+1)
        trial.report(loss_full.detach(), epoch+1)
        print(f'Accuracy {accuracy_full} in epoch {epoch+1}')
    
    trial.set_user_attr("model", cfg['options']['model'])
    trial.set_user_attr("dataset", cfg['dataset']['name'])
    trial.set_user_attr("batchsize", batch_size)
    trial.set_user_attr("accuracy", accuracy_full)
    trial.set_user_attr("f1_score", f1_score_full)
    trial.set_user_attr('bce_loss', loss_full.item())
    trial.set_user_attr('epochs', EPOCHS)
    trial.set_user_attr('reduction_layer', cfg['options']['add_reduction_layer'])

    # Final visualization
    visualization(name_of_the_run, optimizer_name, lr, reduction_val, EPOCHS, model[-1].A.data.clone())


        # Handle pruning based on the intermediate value.
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

    return accuracy_full
