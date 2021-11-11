import numpy as np
import pandas as pd
from typing import Dict, Tuple
from mridle.experiment.experiment import Experiment
from hyperopt import hp


def parse_hyperparams(hyperparams: Dict) -> Dict:
    for entry in hyperparams:
        if isinstance(hyperparams[entry], dict) and len(hyperparams[entry].keys()) == 1:
            solo_key = list(hyperparams[entry].keys())[0]
            if solo_key == 'parse_np_linspace':
                hyperparams[entry] = [int(x) for x in np.linspace(**hyperparams[entry]['parse_np_linspace'])]
            if solo_key == 'parse_hp_choice':
                # parse_hp_choice
                #   name: 'num_estimators'
                #   choice_list: [10, 100, 400]
                hyperparams[entry] = hp.choice(hyperparams[entry]['parse_np_linspace']['name'],
                                               hyperparams[entry]['parse_np_linspace']['choice_list'])

    return hyperparams


def run_experiment(features_df: pd.DataFrame, params: Dict) -> Tuple[Dict, pd.DataFrame]:
    if 'Tuner' in params:
        params['Tuner']['config']['hyperparameters'] = parse_hyperparams(params['Tuner']['config']['hyperparameters'])

    exp = Experiment.configure(params, features_df)
    exp.go()

    serialized_exp = exp.serialize()
    evaluation = exp.evaluation
    return serialized_exp, evaluation
