import unittest
import pandas as pd
import numpy as np
from mridle.pipelines.data_science.feature_engineering.nodes import feature_no_show_before
from mridle.pipelines.data_engineering.ris.nodes import build_status_df, build_slot_df, find_no_shows, \
    set_no_show_severity, STATUS_MAP
from mridle.pipelines.data_science.feature_engineering.nodes import build_feature_set, \
    feature_days_scheduled_in_advance, feature_days_scheduled_in_advance


def day(num_days_from_start, hour=9):
    return pd.Timestamp(year=2019, month=1, day=1, hour=hour, minute=0) + pd.Timedelta(days=num_days_from_start)


code = {status: letter_code for letter_code, status in STATUS_MAP.items()}

valid_date_range = [pd.Timestamp(year=2019, month=1, day=1, hour=0, minute=0),
                    pd.Timestamp(year=2019, month=2, day=1, hour=0, minute=0)]

date_col = 'History_MessageDtTm'
now_status_col = 'History_OrderStatus'
now_sched_for_date_col = 'History_ObsStartPlanDtTm'


class TestNoShowBefore(unittest.TestCase):

    def test_no_show_before_one_case(self):
        slot_df = pd.DataFrame([
            {
                'MRNCmpdId': 1,
                'start_time': str(day(2)),
                'NoShow': 1
            },
            {
                'MRNCmpdId': 1,
                'start_time': str(day(3)),
                'NoShow': 1
            },
            {
                'MRNCmpdId': 1,
                'start_time': str(day(4)),
                'NoShow': 0
            }
        ])

        expected = pd.DataFrame([
            {
                'MRNCmpdId': 1,
                'start_time': str(day(2)),
                'NoShow': 1,
                'no_show_before': 0,
                'no_show_before_sq': 0
            },
            {
                'MRNCmpdId': 1,
                'start_time': str(day(3)),
                'NoShow': 1,
                'no_show_before': 1,
                'no_show_before_sq': 1
            },
            {
                'MRNCmpdId': 1,
                'start_time': str(day(4)),
                'NoShow': 0,
                'no_show_before': 2,
                'no_show_before_sq': 4
            }
        ])
        result = feature_no_show_before(slot_df)

        pd.testing.assert_frame_equal(result, expected, check_like=True)

    def test_no_show_before_no_no_shows(self):
        slot_df = pd.DataFrame([
            {
                'MRNCmpdId': 2,
                'start_time': str(day(2)),
                'NoShow': 0
            },
            {
                'MRNCmpdId': 2,
                'start_time': str(day(3)),
                'NoShow': 0
            }
        ])

        expected = pd.DataFrame([
            {
                'MRNCmpdId': 2,
                'start_time': str(day(2)),
                'NoShow': 0,
                'no_show_before': 0,
                'no_show_before_sq': 0
            },
            {
                'MRNCmpdId': 2,
                'start_time': str(day(3)),
                'NoShow': 0,
                'no_show_before': 0,
                'no_show_before_sq': 0
            }
        ])
        result = feature_no_show_before(slot_df)

        pd.testing.assert_frame_equal(result, expected, check_like=True)

    def test_no_show_before_two_patients(self):
        slot_df = pd.DataFrame([
            {
                'MRNCmpdId': 1,
                'start_time': str(day(2)),
                'NoShow': 1
            },
            {
                'MRNCmpdId': 1,
                'start_time': str(day(3)),
                'NoShow': 1
            },
            {
                'MRNCmpdId': 1,
                'start_time': str(day(4)),
                'NoShow': 0
            },
            {
                'MRNCmpdId': 2,
                'start_time': str(day(2)),
                'NoShow': 0
            },
            {
                'MRNCmpdId': 2,
                'start_time': str(day(3)),
                'NoShow': 0
            }
        ])

        expected = pd.DataFrame([
            {
                'MRNCmpdId': 1,
                'start_time': str(day(2)),
                'NoShow': 1,
                'no_show_before': 0,
                'no_show_before_sq': 0
            },
            {
                'MRNCmpdId': 1,
                'start_time': str(day(3)),
                'NoShow': 1,
                'no_show_before': 1,
                'no_show_before_sq': 1
            },
            {
                'MRNCmpdId': 1,
                'start_time': str(day(4)),
                'NoShow': 0,
                'no_show_before': 2,
                'no_show_before_sq': 4
            },
            {
                'MRNCmpdId': 2,
                'start_time': str(day(2)),
                'NoShow': 0,
                'no_show_before': 0,
                'no_show_before_sq': 0

            },
            {
                'MRNCmpdId': 2,
                'start_time': str(day(3)),
                'NoShow': 0,
                'no_show_before': 0,
                'no_show_before_sq': 0

            }
        ])
        result = feature_no_show_before(slot_df)

        pd.testing.assert_frame_equal(result, expected, check_like=True)

    def test_no_show_before_correct_ordering(self):
        slot_df = pd.DataFrame([
            {
                'MRNCmpdId': 2,
                'start_time': str(day(3)),
                'NoShow': 0
            },
            {
                'MRNCmpdId': 2,
                'start_time': str(day(2)),
                'NoShow': 1
            }
        ])

        expected = pd.DataFrame([
            {
                'MRNCmpdId': 2,
                'start_time': str(day(2)),
                'NoShow': 1,
                'no_show_before': 0,
                'no_show_before_sq': 0

            },
            {
                'MRNCmpdId': 2,
                'start_time': str(day(3)),
                'NoShow': 0,
                'no_show_before': 1,
                'no_show_before_sq': 1

            }
        ])
        result = feature_no_show_before(slot_df)

        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True), check_like=True)


class TestDaysScheduleInAdvance(unittest.TestCase):

    @staticmethod
    def _fill_out_static_columns(raw_df, slot_df, create_fon=True):
        if create_fon:
            raw_df['FillerOrderNo'] = 0
        raw_df['MRNCmpdId'] = '0'
        raw_df['EnteringOrganisationDeviceID'] = 'MR1'
        raw_df['UniversalServiceName'] = 'MR'
        raw_df['OrderStatus'] = raw_df[now_status_col].tail(1).iloc[0]

        if slot_df is not None:
            if create_fon:
                slot_df['FillerOrderNo'] = 0
            slot_df['MRNCmpdId'] = '0'
            slot_df['EnteringOrganisationDeviceID'] = 'MR1'
            slot_df['UniversalServiceName'] = 'MR'

            slot_df_col_order = ['FillerOrderNo',
                                 'MRNCmpdId',
                                 'patient_class_adj',
                                 'start_time',
                                 'end_time',
                                 'NoShow',
                                 'slot_outcome',
                                 'slot_type',
                                 'slot_type_detailed',
                                 'EnteringOrganisationDeviceID',
                                 'UniversalServiceName',
                                 'sched_days_advanced',
                                 'sched_days_advanced_busday',
                                 'sched_days_advanced_sq',
                                 'sched_2_days'
                                 ]
            slot_df = slot_df[slot_df_col_order]

        return raw_df, slot_df

    def test_basic_sched_days_advanced(self):
        raw_df = pd.DataFrame.from_records([
            # date,                               now_status,            now_sched_for_date
            (day(0), code['scheduled'], day(4)),
            (day(4), code['started'], day(4)),
            (day(4) + pd.Timedelta(minutes=30), code['examined'], day(4)),
        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'ambulant'

        expected_slot_df = pd.DataFrame([
            {
                'start_time': day(4),
                'end_time': day(4) + pd.Timedelta(minutes=30),
                'NoShow': False,
                'slot_outcome': 'show',
                'slot_type': 'show',
                'slot_type_detailed': 'show',
                'duplicate_appt': 1,
                'patient_class_adj': 'ambulant',
                'sched_days_advanced': 4,
                'sched_days_advanced_busday': 4,
                'sched_days_advanced_sq': 16,
                'sched_2_days': False
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, valid_date_range)
        slot_df_with_feature = feature_days_scheduled_in_advance(status_df, slot_df)

        pd.testing.assert_frame_equal(slot_df_with_feature, expected_slot_df, check_like=True)

    def test_sched_days_advanced_with_multiple_changes(self):
        raw_df = pd.DataFrame.from_records([
            # date,                               now_status,            now_sched_for_date
            (day(0), code['requested'], day(0)),
            (day(0), code['scheduled'], day(10)),
            (day(1), code['scheduled'], day(14)),
            (day(14), code['started'], day(14)),
            (day(14) + pd.Timedelta(minutes=30), code['examined'], day(14)),
        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'ambulant'

        expected_slot_df = pd.DataFrame([
            {
                'start_time': day(14),
                'end_time': day(14) + pd.Timedelta(minutes=30),
                'NoShow': False,
                'slot_outcome': 'show',
                'slot_type': 'show',
                'slot_type_detailed': 'show',
                'duplicate_appt': 1,
                'patient_class_adj': 'ambulant',
                'sched_days_advanced': 13,
                'sched_days_advanced_busday': 9,
                'sched_days_advanced_sq': 169,
                'sched_2_days': False
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, valid_date_range)
        slot_df_with_feature = feature_days_scheduled_in_advance(status_df, slot_df)

        pd.testing.assert_frame_equal(slot_df_with_feature, expected_slot_df, check_like=True)

    def test_sched_days_advanced_with_one_noshow_one_ok_appt(self):
        raw_df = pd.DataFrame.from_records([
            # date,                               now_status,            now_sched_for_date
            (day(0), code['requested'], day(0)),
            (day(0), code['scheduled'], day(10)),
            (day(9), code['scheduled'], day(14)),
            (day(14), code['started'], day(14)),
            (day(14) + pd.Timedelta(minutes=30), code['examined'], day(14)),
        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'ambulant'

        expected_slot_df = pd.DataFrame([
            {
                'start_time': day(10),
                'end_time': day(10) + pd.Timedelta(minutes=30),
                'NoShow': True,
                'slot_outcome': 'rescheduled',
                'slot_type': 'no-show',
                'slot_type_detailed': 'soft no-show',
                'duplicate_appt': 1,
                'patient_class_adj': 'ambulant',
                'sched_days_advanced': 10,
                'sched_days_advanced_busday': 8,
                'sched_days_advanced_sq': 100,
                'sched_2_days': False

            },
            {
                'start_time': day(14),
                'end_time': day(14) + pd.Timedelta(minutes=30),
                'NoShow': False,
                'slot_outcome': 'show',
                'slot_type': 'show',
                'slot_type_detailed': 'show',
                'duplicate_appt': 1,
                'patient_class_adj': 'ambulant',
                'sched_days_advanced': 5,
                'sched_days_advanced_busday': 3,
                'sched_days_advanced_sq': 25,
                'sched_2_days': False

            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, valid_date_range)
        slot_df_with_feature = feature_days_scheduled_in_advance(status_df, slot_df)

        pd.testing.assert_frame_equal(slot_df_with_feature, expected_slot_df, check_like=True)

    def test_sched_days_advanced_future(self):
        raw_df = pd.DataFrame.from_records([
            # date,                               now_status,            now_sched_for_date
            (day(0), code['requested'], day(0)),
            (day(0), code['scheduled'], day(10)),
            (day(1), code['scheduled'], day(14)),
        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'ambulant'

        expected_slot_df = pd.DataFrame([
            {
                'start_time': day(14),
                'end_time': day(14) + pd.Timedelta(minutes=30),
                'NoShow': False,
                'slot_outcome': None,
                'slot_type': None,
                'slot_type_detailed': None,
                'duplicate_appt': 1,
                'patient_class_adj': 'ambulant',
                'sched_days_advanced': 13,
                'sched_days_advanced_busday': 9,
                'sched_days_advanced_sq': 169,
                'sched_2_days': False
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, valid_date_range, build_future_slots=True)
        slot_df_with_feature = feature_days_scheduled_in_advance(status_df, slot_df)

        pd.testing.assert_frame_equal(slot_df_with_feature, expected_slot_df, check_like=True)
