import numpy as np
import unittest
from mridle.utilities.modeling import parse_hyperparams


class TestParseHyperParams(unittest.TestCase):

    def test_parse_start_stop_num_to_linspace(self):
        test_input = {
            'n_estimators': {
                'parse_np_linspace': {
                    'start': 200,
                    'stop': 2000,
                    'num': 10,
                }
            }
        }
        expected_result = {
            'n_estimators': list(np.linspace(200, 2000, num=10))
        }
        test_result = parse_hyperparams(test_input)
        self.assertEqual(test_result, expected_result)

    def test_parse_list_no_change(self):
        test_input = {
            'max_features': ['auto', 'sqrt']
        }
        expected_result = {
            'max_features': ['auto', 'sqrt']
        }
        test_result = parse_hyperparams(test_input)
        self.assertEqual(test_result, expected_result)



import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
from torch import nn
import torch
from skorch import NeuralNet
from sklearn.preprocessing import FunctionTransformer, StandardScaler, LabelEncoder


column_names = ['A', 'B', 'C', 'D']
df = pd.DataFrame(np.random.randint(0, 100, size=(100, len(column_names))), columns=column_names)
df['E'] = random.choices(['one', 'two', 'three'], k=100)
df['label'] = np.where(df[column_names[0]] > 50, 1, 0)

categorical_columns = ['E']
numerical_columns = ['A', 'B', 'C', 'D']

categorical_encoder = OneHotEncoder(handle_unknown='ignore')

numerical_pipe = Pipeline([
    ('scaler', StandardScaler())
])

preprocessing = ColumnTransformer(
    [('cat', categorical_encoder, categorical_columns),
     ('num', numerical_pipe, numerical_columns)
    ]
)


class MLP(nn.Module):
    def __init__(self, input_layer_size, hidden_layer_size=24, dropout_p=0.1):
        super().__init__()

        self.fc1 = nn.Linear(input_layer_size, hidden_layer_size)
        self.relu = torch.nn.ReLU()
        # self.fc2 = nn.Linear(hidden_layer_size, int(hidden_layer_size/2))
        self.fc3 = torch.nn.Linear(hidden_layer_size, 1)
        self.out = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # x = self.fc2(x)
        # x = self.relu(x)

        x = self.dropout(x)
        x = self.fc3(x)
        output = self.out(x)
        return output
