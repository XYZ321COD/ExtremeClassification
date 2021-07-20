from utils.setup_dataset import get_train_dataset, get_eval_datasets
from utils.setup_model import get_model
from utils.metics import MCELoss, ECELoss
import torch.nn as nn
import torch.optim as optim
import torch
import yaml
import sklearn.metrics as mt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from visual import visualization
import torch.nn.utils.prune as prune

file = open(r'config.yaml')
cfg = yaml.load(file, Loader=yaml.FullLoader)


def objective(trial, name_of_the_run=cfg['options']['name_of_the_run']):
    
    DEVICE = torch.device(cfg['options']['device'])
    EPOCHS = cfg['options']['epochs']

    model, reduction_val = get_model(trial=trial)
    model.to(DEVICE)

    optimizer_name = trial.suggest_categorical("optimizer", cfg['hyperparameters']['optimizers'])
    lr = trial.suggest_float("lr", min(cfg['hyperparameters']['lr']), max(cfg['hyperparameters']['lr']))
    prunning_value = trial.suggest_float("pr_val", min(cfg['hyperparameters']['prunning_val']), max(cfg['hyperparameters']['prunning_val']))
    WRITTER = SummaryWriter('{}/{}_{}_red_{}_pr{}'.format(name_of_the_run, optimizer_name, lr, reduction_val, prunning_value))

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    ece_loss = ECELoss()
    mce_loss = MCELoss()

    (train_loader, valid_loader), batch_size = get_train_dataset(trial=trial)
    eval_dataset, _ = get_eval_datasets(trial)
    # TODO change for bigger number of datasets
    eval_loader = eval_dataset[0][0]
    
    # Get the W to visualize
    # W = list(model.children())[-2][-1].weight
    # visualization(name_of_the_run, optimizer_name, lr, reduction_val, 0, W, prunning_value)
    
    for epoch in range(EPOCHS):
        model.train()
        for (data, target) in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            target = nn.functional.one_hot(target, num_classes=10).to(dtype=torch.float32)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.BCELoss(reduction='sum')(output, target)
            loss.backward()
            optimizer.step()
        if epoch == 5:
            prune.l1_unstructured(list(model.children())[-2][-1], name='weight', amount=prunning_value)
        # Validation of the model.
        model.eval()
        accuracy = f1_score = loss = ece = mce = 0
        with torch.no_grad():
            for (data, target) in valid_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                target_ = target.detach().clone()
                target = nn.functional.one_hot(target, num_classes=10).to(dtype=torch.float32)
                loss += nn.BCELoss(reduction='sum')(output, target)
                accuracy += mt.accuracy_score(target.cpu().detach(), output.cpu().detach() > cfg['options']['threshold'])
                f1_score += mt.f1_score(target.cpu().detach(), output.cpu().detach() > cfg['options']['threshold'], average="samples")

                ece += ece_loss(output.cpu().numpy(), target_.cpu().numpy())
                mce += mce_loss(output.cpu().numpy(), target_.cpu().numpy())

        accuracy_full = accuracy / len(valid_loader)
        f1_score_full = f1_score / len(valid_loader)
        loss_full = loss / len(valid_loader)
        ece_full = ece / len(valid_loader)
        mce_full = mce / len(valid_loader)

        for n, p in model.named_parameters():
            if 'bias' not in n:
                WRITTER.add_histogram('{}'.format(n), p, epoch)
                if p.requires_grad:
                    WRITTER.add_histogram('{}.grad'.format(n), p.grad, epoch)
        WRITTER.add_scalar('BCE Loss', (loss_full.item()), epoch+1)
        WRITTER.add_scalar('Acc', accuracy_full, epoch+1)
        trial.report(accuracy_full, epoch+1)
        trial.report(f1_score_full, epoch+1)
        trial.report(loss_full.detach(), epoch+1)
        trial.report(ece_full, epoch+1)
        trial.report(mce_full, epoch+1)
        print(f'Accuracy {accuracy_full} in epoch {epoch+1} | ECE {ece_full} | MCE {mce_full}')


    # Eval on EvalDATASET
    model.eval()
    accuracy = f1_score = ece = mce = 0
    with torch.no_grad():
        for (data, target) in eval_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            target_ = target.detach().clone()
            target = nn.functional.one_hot(target, num_classes=10).to(dtype=torch.float32)
            accuracy += mt.accuracy_score(target.cpu().detach(), output.cpu().detach() > cfg['options']['threshold'])
            f1_score += mt.f1_score(target.cpu().detach(), output.cpu().detach() > cfg['options']['threshold'], average="samples")
            ece += ece_loss(output.cpu().numpy(), target_.cpu().numpy())
            mce += mce_loss(output.cpu().numpy(), target_.cpu().numpy())

    eval_accuracy_full = accuracy / len(eval_loader)
    eval_f1_score_full = f1_score / len(eval_loader)
    eval_ece_full = ece / len(eval_loader)
    eval_mce_full = mce / len(eval_loader)

    print(f'Eval accuracy {eval_accuracy_full} | Eval ECE {eval_ece_full} | Eval MCE {eval_mce_full}')

    trial.set_user_attr("model", cfg['options']['model'])
    trial.set_user_attr("dataset", cfg['dataset']['name'])
    trial.set_user_attr("batchsize", batch_size)
    trial.set_user_attr("accuracy", accuracy_full)
    trial.set_user_attr("f1_score", f1_score_full)
    trial.set_user_attr('bce_loss', loss_full.item())

    trial.set_user_attr("train_MCE", mce_full)
    trial.set_user_attr("train_ECE", ece_full)
    trial.set_user_attr("eval_accuracy", eval_accuracy_full)
    trial.set_user_attr("eval_f1_score", f1_score_full)
    trial.set_user_attr("eval_MCE", eval_mce_full)
    trial.set_user_attr("eval_ECE", eval_ece_full)

    trial.set_user_attr('epochs', EPOCHS)
    trial.set_user_attr('reduction_layer', cfg['options']['add_reduction_layer'])

    return accuracy_full
