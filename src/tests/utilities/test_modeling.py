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


x_train = df.copy().drop('label', axis=1)
y = df['label']

n_cols = preprocessing.fit_transform(x_train).shape[1]

net = NeuralNet(
    MLP(input_layer_size=n_cols, hidden_layer_size=20, dropout_p=0),
    criterion=nn.BCELoss,
    # criterion__weight=torch.tensor(0.17),
    lr=0.01,
    optimizer=torch.optim.SGD,
    batch_size=32,
    max_epochs=100,
    verbose=0,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
)


def tofloat32(x):
    return x.astype(np.float32)


def transform_y(y):
        y = LabelEncoder().fit_transform(y)
        y = y.astype('float32')
        y = y.reshape((len(y), 1))
        return y


nn_pipe = Pipeline([
    ('preprocess', preprocessing),
    ('tofloat32', FunctionTransformer(tofloat32, accept_sparse=True)),
    ('classifier', net)
])

nn_pipe.fit(x_train, transform_y(y))

nn_pipe.predict(x_train)
