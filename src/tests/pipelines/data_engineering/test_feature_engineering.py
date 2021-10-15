import unittest
import pandas as pd
import numpy as np
from mridle.pipelines.data_science.feature_engineering.nodes import feature_no_show_before
def day(num_days_from_start, hour=9):
    return pd.Timestamp(year=2019, month=1, day=1, hour=hour, minute=0) + pd.Timedelta(days=num_days_from_start)


class TestFeatureEngineering(unittest.TestCase):

    def test_no_show_before(self):
        """
        3 test examples included here.

        First one with MRNCmpdID=1 is the case where the patient had three appointments, and the first two were no-shows
        Second is with MRNCmpdID=2 is the case where the patient showed up for their first appt, but not their second -
        both these rows should be returned with no_show_before equal to 0.
        The case with MRNCmpdID=3 is where the patient alternates shows and no-shows
        """
        slot_df = pd.DataFrame([
            {
                'MRNCmpdId': 1,
                'date': str(day(2).date()),
                'start_time': str(day(3).time()),
                'NoShow': 1
            },
            {
                'MRNCmpdId': 1,
                'date': str(day(3).date()),
                'start_time': str(day(3).time()),
                'NoShow': 1
            },
            {
                'MRNCmpdId': 1,
                'date': str(day(4).date()),
                'start_time': str(day(3).time()),
                'NoShow': 0
            },
            {
                'MRNCmpdId': 2,
                'date': str(day(2).date()),
                'start_time': str(day(3).time()),
                'NoShow': 0
            },
            {
                'MRNCmpdId': 2,
                'date': str(day(3).date()),
                'start_time': str(day(3).time()),
                'NoShow': 1
            },
            {
                'MRNCmpdId': 3,
                'date': str(day(1).date()),
                'start_time': str(day(3).time()),
                'NoShow': 1
            },
            {
                'MRNCmpdId': 3,
                'date': str(day(2).date()),
                'start_time': str(day(3).time()),
                'NoShow': 0
            },
            {
                'MRNCmpdId': 3,
                'date': str(day(3).date()),
                'start_time': str(day(3).time()),
                'NoShow': 1
            },
            {
                'MRNCmpdId': 3,
                'date': str(day(4).date()),
                'start_time': str(day(3).time()),
                'NoShow': 0
            }
        ])

        expected = pd.DataFrame([
            {
                'MRNCmpdId': 1,
                'date': str(day(2).date()),
                'start_time': str(day(3).time()),
                'NoShow': 1,
                'no_show_before': 0
            },
            {
                'MRNCmpdId': 1,
                'date': str(day(3).date()),
                'start_time': str(day(3).time()),
                'NoShow': 1,
                'no_show_before': 1
            },
            {
                'MRNCmpdId': 1,
                'date': str(day(4).date()),
                'start_time': str(day(3).time()),
                'NoShow': 0,
                'no_show_before': 2
            },
            {
                'MRNCmpdId': 2,
                'date': str(day(2).date()),
                'start_time': str(day(3).time()),
                'NoShow': 0,
                'no_show_before': 0
            },
            {
                'MRNCmpdId': 2,
                'date': str(day(3).date()),
                'start_time': str(day(3).time()),
                'NoShow': 1,
                'no_show_before': 0
            },
            {
                'MRNCmpdId': 3,
                'date': str(day(1).date()),
                'start_time': str(day(3).time()),
                'NoShow': 1,
                'no_show_before': 0
            },
            {
                'MRNCmpdId': 3,
                'date': str(day(2).date()),
                'start_time': str(day(3).time()),
                'NoShow': 0,
                'no_show_before': 1
            },
            {
                'MRNCmpdId': 3,
                'date': str(day(3).date()),
                'start_time': str(day(3).time()),
                'NoShow': 1,
                'no_show_before': 1
            },
            {
                'MRNCmpdId': 3,
                'date': str(day(4).date()),
                'start_time': str(day(3).time()),
                'NoShow': 0,
                'no_show_before': 2
            }
        ])
        result = feature_no_show_before(slot_df)

        pd.testing.assert_frame_equal(result, expected, check_like=True)



