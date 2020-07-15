import unittest
import pandas as pd
import numpy as np
from mridle.data_management import build_slot_df


def day(num_days_from_start):
    return pd.Timestamp(year=2019, month=1, day=1, hour=9, minute=0) + pd.Timedelta(days=num_days_from_start)


class TestBuildSlotDF(unittest.TestCase):

    def test_basic(self):
        status_df = pd.DataFrame([
            {
                'date': day(0),
                'now_status': 'scheduled',
                'was_sched_for_date': day(4),
                'NoShow': False,
                'NoShow_outcome': np.NaN,
                'slot_type': 'show',
                'slot_type_detailed': 'show',
            },
            {
                'date': day(4),
                'now_status': 'started',
                'was_sched_for_date': day(4),
                'NoShow': False,
                'NoShow_outcome': np.NaN,
                'slot_type': 'show',
                'slot_type_detailed': 'show',
            },
            {
                'date': day(4) + pd.Timedelta(minutes=30),
                'now_status': 'examined',
                'was_sched_for_date': day(4),
                'NoShow': False,
                'NoShow_outcome': np.NaN,
                'slot_type': 'show',
                'slot_type_detailed': 'show',
            },
        ])
        expected_slot_df = pd.DataFrame([
            {
                'FillerOrderNo': 0,
                'start_time': day(4),
                'end_time': day(4) + pd.Timedelta(minutes=30),
                'NoShow': False,
                'NoShow_outcome': np.NaN,
                'slot_type': 'show',
                'slot_type_detailed': 'show',
                'EnteringOrganisationDeviceID': 'MR1',
                'UniversalServiceName': 'MR',
            }
        ])
        status_df['FillerOrderNo'] = 0
        status_df['EnteringOrganisationDeviceID'] = 'MR1'
        status_df['UniversalServiceName'] = 'MR'

        slot_df = build_slot_df(status_df)

        pd.testing.assert_frame_equal(slot_df, expected_slot_df, check_like=True)

    def test_series_of_scheduling_changes_no_slot(self):
        status_df = pd.DataFrame([
            {
                'date': day(0),
                'now_status': 'scheduled',
                'was_sched_for_date': np.NaN,
                'now_sched_for_date': day(7),
                'NoShow': False,
                'NoShow_outcome': np.NaN,
                'slot_type': 'show',
                'slot_type_detailed': 'show',
            },
            {
                'date': day(1),
                'now_status': 'scheduled',
                'was_sched_for_date': day(7),
                'now_sched_for_date': day(14),
                'NoShow': False,
                'NoShow_outcome': np.NaN,
                'slot_type': 'show',
                'slot_type_detailed': 'show',
            },
            {
                'date': day(2),
                'now_status': 'scheduled',
                'was_sched_for_date': day(14),
                'now_sched_for_date': day(21),
                'NoShow': False,
                'NoShow_outcome': np.NaN,
                'slot_type': 'show',
                'slot_type_detailed': 'show',
            },
        ])

        status_df['FillerOrderNo'] = 0
        status_df['EnteringOrganisationDeviceID'] = 'MR1'
        status_df['UniversalServiceName'] = 'MR'

        slot_df = build_slot_df(status_df)

        self.assertEqual(slot_df.shape[0], 0)


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

