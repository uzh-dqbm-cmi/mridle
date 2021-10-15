import unittest
import pandas as pd
import numpy as np
from mridle.pipelines.data_science.feature_engineering.nodes import feature_no_show_before
def day(num_days_from_start, hour=9):
    return pd.Timestamp(year=2019, month=1, day=1, hour=hour, minute=0) + pd.Timedelta(days=num_days_from_start)


class TestFeatureEngineering(unittest.TestCase):

    def test_no_show_before_one_case(self):
        slot_df = pd.DataFrame([
            {
                'MRNCmpdId': 1,
                'date': str(day(2)),
                'NoShow': 1
            },
            {
                'MRNCmpdId': 1,
                'date': str(day(3)),
                'NoShow': 1
            },
            {
                'MRNCmpdId': 1,
                'date': str(day(4)),
                'NoShow': 0
            }
        ])

        expected = pd.DataFrame([
            {
                'MRNCmpdId': 1,
                'date': str(day(2)),
                'NoShow': 1,
                'no_show_before': 0
            },
            {
                'MRNCmpdId': 1,
                'date': str(day(3)),
                'NoShow': 1,
                'no_show_before': 1
            },
            {
                'MRNCmpdId': 1,
                'date': str(day(4)),
                'NoShow': 0,
                'no_show_before': 2
            }
        ])
        result = feature_no_show_before(slot_df)

        pd.testing.assert_frame_equal(result, expected, check_like=True)

    def test_no_show_before_no_no_shows(self):
        slot_df = pd.DataFrame([
            {
                'MRNCmpdId': 2,
                'date': str(day(2)),
                'NoShow': 0
            },
            {
                'MRNCmpdId': 2,
                'date': str(day(3)),
                'NoShow': 0
            }
        ])

        expected = pd.DataFrame([
            {
                'MRNCmpdId': 2,
                'date': str(day(2)),
                'NoShow': 0,
                'no_show_before': 0
            },
            {
                'MRNCmpdId': 2,
                'date': str(day(3)),
                'NoShow': 0,
                'no_show_before': 0
            }
        ])
        result = feature_no_show_before(slot_df)

        pd.testing.assert_frame_equal(result, expected, check_like=True)

    def test_no_show_before_two_patients(self):
        slot_df = pd.DataFrame([
            {
                'MRNCmpdId': 1,
                'date': str(day(2)),
                'NoShow': 1
            },
            {
                'MRNCmpdId': 1,
                'date': str(day(3)),
                'NoShow': 1
            },
            {
                'MRNCmpdId': 1,
                'date': str(day(4)),
                'NoShow': 0
            },
            {
                'MRNCmpdId': 2,
                'date': str(day(2)),
                'NoShow': 0
            },
            {
                'MRNCmpdId': 2,
                'date': str(day(3)),
                'NoShow': 0
            }
        ])

        expected = pd.DataFrame([
            {
                'MRNCmpdId': 1,
                'date': str(day(2)),
                'NoShow': 1,
                'no_show_before': 0
            },
            {
                'MRNCmpdId': 1,
                'date': str(day(3)),
                'NoShow': 1,
                'no_show_before': 1
            },
            {
                'MRNCmpdId': 1,
                'date': str(day(4)),
                'NoShow': 0,
                'no_show_before': 2
            },
            {
                'MRNCmpdId': 2,
                'date': str(day(2)),
                'NoShow': 0,
                'no_show_before': 0
            },
            {
                'MRNCmpdId': 2,
                'date': str(day(3)),
                'NoShow': 0,
                'no_show_before': 0
            }
        ])
        result = feature_no_show_before(slot_df)

        pd.testing.assert_frame_equal(result, expected, check_like=True)

    def test_no_show_before_correct_ordering(self):
        slot_df = pd.DataFrame([
            {
                'MRNCmpdId': 2,
                'date': str(day(3)),
                'NoShow': 0
            },
            {
                'MRNCmpdId': 2,
                'date': str(day(2)),
                'NoShow': 1
            }
        ])

        expected = pd.DataFrame([
            {
                'MRNCmpdId': 2,
                'date': str(day(2)),
                'NoShow': 1,
                'no_show_before': 0
            },
            {
                'MRNCmpdId': 2,
                'date': str(day(3)),
                'NoShow': 0,
                'no_show_before': 1
            }
        ])
        result = feature_no_show_before(slot_df)

        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True), check_like=True)

