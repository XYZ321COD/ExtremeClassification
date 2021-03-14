from utils import get_dataset
import torch.nn as nn
import torch.optim as optim
import torch
import yaml
from model import define_model, add_aggregation_to_model
import sklearn.metrics as mt


def objective(trial):

    file = open(r'config.yaml')
    cfg = yaml.load(file, Loader=yaml.FullLoader)

    DEVICE = torch.device(cfg['options']['device'])
    BATCHSIZE = cfg['options']['batch_size']
    EPOCHS = cfg['options']['epochs']
    N_TRAIN_EXAMPLES = BATCHSIZE * 120
    N_VALID_EXAMPLES = BATCHSIZE * 40

    # Generate the model.
    model = define_model().to(DEVICE)
    # Add agregation layer
    model = add_aggregation_to_model(model, 10, 10)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", cfg['hyperparameters']['optimizers'])
    lr = trial.suggest_categorical("lr", cfg['hyperparameters']['lr'])

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the MNIST dataset.
    train_loader, valid_loader = get_dataset.get_mnist()

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.to(DEVICE), target.to(DEVICE)
            target = nn.functional.one_hot(target, num_classes=10).to(dtype=torch.float32)
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
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                target = nn.functional.one_hot(target, num_classes=cfg['options']['num_classes']).to(dtype=torch.float32)
                loss += nn.BCELoss(reduction='sum')(output, target)
                accuracy += mt.accuracy_score(target.cpu().detach(), output.cpu().detach() > cfg['options']['threshold'])
                f1_score += mt.f1_score(target.cpu().detach(), output.cpu().detach() > cfg['options']['threshold'], average="samples")

        accuracy_full = accuracy / min(len(valid_loader.dataset), N_VALID_EXAMPLES / BATCHSIZE)
        f1_score_full = f1_score / min(len(valid_loader.dataset), N_VALID_EXAMPLES / BATCHSIZE)
        loss_full = loss / min(len(valid_loader.dataset), N_VALID_EXAMPLES / BATCHSIZE)
        trial.report(accuracy_full, epoch)
        trial.report(f1_score_full, epoch)
        trial.report(loss_full, epoch)
    
    trial.set_user_attr("accuracy", accuracy_full)
    trial.set_user_attr("f1_score", f1_score_full)
    trial.set_user_attr('bce_loss', loss_full)


        # Handle pruning based on the intermediate value.
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

    return accuracy_full
