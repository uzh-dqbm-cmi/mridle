import unittest
import pandas as pd
import numpy as np
from mridle.data_management import build_slot_df


class TestBuildSlotDF(unittest.TestCase):

    def custom_assert_df_equal(self, df, expected_df):
        for i in range(len(expected_df)):
            df_dict = dict(df.loc[i])
            expected_df_dict = dict(expected_df.loc[i])
            for key in expected_df_dict:
                self.assertEqual(df_dict[key], expected_df_dict[key])

    def test_basic(self):
        status_df = pd.DataFrame([
            {
                'FillerOrderNo': 0,
                'date': pd.Timestamp(year=2019, month=1, day=1, hour=9, minute=0),
                'now_status': 'started',
                'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9, minute=0),
                'NoShow': False,
                'NoShow_outcome': 'nan',  # TODO: make real nan like in real code
                'slot_type': 'show',
                'slot_type_detailed': 'show',
                'EnteringOrganisationDeviceID': 'MR1',
                'UniversalServiceName': 'MR',
            },
            {
                'FillerOrderNo': 0,
                'date': pd.Timestamp(year=2019, month=1, day=1, hour=9, minute=30),
                'now_status': 'examined',
                'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9, minute=0),
                'NoShow': False,
                'NoShow_outcome': 'nan',  # TODO: make real nan like in real code
                'slot_type': 'show',
                'slot_type_detailed': 'show',
                'EnteringOrganisationDeviceID': 'MR1',
                'UniversalServiceName': 'MR',
            },
        ])
        expected_slot_df = pd.DataFrame([
            {
                'FillerOrderNo': 0.0,  # TODO: why does this change to float?!
                'start_time': pd.Timestamp(year=2019, month=1, day=1, hour=9, minute=0),
                'end_time': pd.Timestamp(year=2019, month=1, day=1, hour=9, minute=30),
                'NoShow': False,
                'NoShow_outcome': 'nan',  # TODO: make real nan like in real code
                'slot_type': 'show',
                'slot_type_detailed': 'show',
                'EnteringOrganisationDeviceID': 'MR1',
                'UniversalServiceName': 'MR',
            }
        ])
        slot_df = build_slot_df(status_df)

        self.custom_assert_df_equal(slot_df, expected_slot_df)

        # TODO: I'm pretty sure there is a pandas dataframe assertEquals
        #  ... maybe also with non-exact values for the float issue? though not a good solution
        # self.assertEqual(build_slot_df(status_df), expected_slot_df)

    def test_MRNCmpdId_not_included(self):
        return True

    def test_not_include_id_cols(self):
        return True

    def test_time_of_examined_not_equals_scheduled_plus_30(self):
        return True

    def test_soft_noshow(self):
        return True

    def test_hard_noshow(self):
        return True

    def test_not_a_slot(self):
        return True

