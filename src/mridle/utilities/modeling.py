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
            elif solo_key == 'parse_hp_choice':
                # parse_hp_choice
                #   choice_list: [10, 100, 400]
                if 'choice_list' in hyperparams[entry][solo_key]:
                    hyperparams[entry] = hp.choice(entry, hyperparams[entry][solo_key]['choice_list'])
                else:
                    raise ValueError('parse_hp_choice requires ''choice_list'' as a parameter in the config file')

            elif solo_key == 'parse_hp_uniform':  # Uniform within range
                # parse_hp_uniform
                #   start: 10
                #   end: 20
                if all(x in hyperparams[entry][solo_key] for x in ['start', 'end']):
                    hyperparams[entry] = hp.uniform(entry, hyperparams[entry][solo_key]['start'],
                                                    hyperparams[entry][solo_key]['end'])
                else:
                    raise ValueError('parse_hp_uniform requires ''start'' and ''end'' as parameters in the config file')

            elif solo_key == 'parse_hp_quniform':  # Uniform over integers within range
                # parse_hp_quniform
                #   start: 10
                #   end: 20
                if all(x in hyperparams[entry][solo_key] for x in ['start', 'end']):
                    hyperparams[entry] = hp.quniform(entry, hyperparams[entry][solo_key]['start'],
                                                     hyperparams[entry][solo_key]['end'])
                else:
                    raise ValueError('parse_hp_quniform requires ''start'' and ''end'' as parameters in the config '
                                     'file')

            elif solo_key == 'parse_hp_loguniform':
                # parse_hp_loguniform
                #   start: 0
                #   end: 1
                if all(x in hyperparams[entry][solo_key] for x in ['start', 'end']):
                    hyperparams[entry] = hp.loguniform(entry, hyperparams[entry][solo_key]['start'],
                                                       hyperparams[entry][solo_key]['end'])
                else:
                    raise ValueError('parse_hp_loguniform requires ''start'' and ''end'' as parameters in '
                                     'the config file')

    return hyperparams


def run_experiment(features_df: pd.DataFrame, params: Dict) -> Tuple[Dict, pd.DataFrame]:
    if 'Tuner' in params:
        params['Tuner']['config']['hyperparameters'] = parse_hyperparams(params['Tuner']['config']['hyperparameters'])

    exp = Experiment.configure(params, features_df)
    exp.go()

    serialized_exp = exp.serialize()
    evaluation = exp.evaluation
    return serialized_exp, evaluation
