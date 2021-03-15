import optuna
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history
from trainer import objective
from datetime import datetime
import matplotlib.pyplot as plt
import yaml

def experiments():
    file = open(r'config.yaml')
    cfg = yaml.load(file, Loader=yaml.FullLoader)

    study = optuna.create_study(direction="maximize")
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
    dataframe.to_csv("output-{}.csv".format(datetime.now()))

if __name__ == "__main__":
    experiments()