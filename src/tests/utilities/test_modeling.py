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
