import optuna
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history
from optuna.samplers import GridSampler
from trainer import objective
from datetime import datetime
import matplotlib.pyplot as plt
import yaml

def experiments():
    file = open(r'config.yaml')
    cfg = yaml.load(file, Loader=yaml.FullLoader)

    search_space = {"lr": cfg['hyperparameters']['lr'], "optimizer": cfg['hyperparameters']['optimizers'],
         "batchsize" : cfg['hyperparameters']['batchsize'], "reduction_value": cfg['hyperparameters']['reduction_value']}

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.GridSampler(search_space))

    study.optimize(objective, n_trials=cfg['options']['number_of_trails'], timeout=6000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.show()
    dataframe = study.trials_dataframe()
    dataframe = dataframe .sort_values(by=["params_reduction_value"], ascending=False)
    dataframe.to_csv("output-{}.csv".format(cfg['options']['name_of_the_run']))

if __name__ == "__main__":
    experiments()