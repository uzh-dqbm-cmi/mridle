import unittest
import pandas as pd
import numpy as np
from mridle.data_management import build_status_df, build_slot_df, STATUS_MAP


code = {status: letter_code for letter_code, status in STATUS_MAP.items()}


def day(num_days_from_start):
    return pd.Timestamp(year=2019, month=1, day=1, hour=9, minute=0) + pd.Timedelta(days=num_days_from_start)


date_col = 'History_MessageDtTm'
now_status_col = 'History_OrderStatus'
now_sched_for_date_col = 'History_ObsStartPlanDtTm'


class TestBuildSlotDF(unittest.TestCase):

    @staticmethod
    def _fill_out_static_columns(raw_df, slot_df):
        raw_df['FillerOrderNo'] = 0
        raw_df['MRNCmpdId'] = '0'
        raw_df['EnteringOrganisationDeviceID'] = 'MR1'
        raw_df['UniversalServiceName'] = 'MR'
        raw_df['OrderStatus'] = raw_df[now_status_col].tail(1).iloc[0]

        if slot_df is not None:
            slot_df['FillerOrderNo'] = 0
            slot_df['MRNCmpdId'] = '0'
            slot_df['EnteringOrganisationDeviceID'] = 'MR1'
            slot_df['UniversalServiceName'] = 'MR'

            slot_df_col_order = ['FillerOrderNo',
                                 'MRNCmpdId',
                                 'start_time',
                                 'end_time',
                                 'NoShow',
                                 'slot_outcome',
                                 'slot_type',
                                 'slot_type_detailed',
                                 'EnteringOrganisationDeviceID',
                                 'UniversalServiceName',
                                 'time_slot_status',
                                 'duplicate_appt',
                                 ]
            slot_df = slot_df[slot_df_col_order]

        return raw_df, slot_df

    def test_basic_show(self):
        raw_df = pd.DataFrame.from_records([
                # date,                               now_status,            now_sched_for_date
                (day(0),                              code['scheduled'],     day(4)),
                (day(4),                              code['started'],       day(4)),
                (day(4) + pd.Timedelta(minutes=30),   code['examined'],      day(4)),
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
                'time_slot_status': False,
                'duplicate_appt': 1,
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df)
        slot_df = build_slot_df(status_df, include_id_cols=True)

        pd.testing.assert_frame_equal(slot_df, expected_slot_df, check_like=True)

    def test_series_of_scheduling_changes_no_slot(self):
        raw_df = pd.DataFrame.from_records([
                # date,     now_status,            now_sched_for_date
                (day(0),    code['scheduled'],     day(7)),
                (day(1),    code['scheduled'],     day(14)),
                (day(2),    code['scheduled'],     day(21)),
            ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'ambulant'

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, None)
        status_df = build_status_df(raw_df)
        slot_df = build_slot_df(status_df)

        self.assertEqual(slot_df.shape[0], 0)

    def test_soft_noshow(self):
        raw_df = pd.DataFrame.from_records([
            # date,    now_status,           now_sched_for_date
            (day(0),   code['scheduled'],    day(7)),
            (day(6),   code['scheduled'],    day(14)),
            (day(13),  code['registered'],   day(14)),
            (day(14),  code['started'],      day(14)),
            (day(14),  code['examined'],     day(14)),
        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'ambulant'

        expected_slot_df = pd.DataFrame([
            {
                'start_time': day(7),
                'end_time': day(7) + pd.Timedelta(minutes=30),
                'NoShow': True,
                'slot_outcome': 'rescheduled',
                'slot_type': 'no-show',
                'slot_type_detailed': 'soft no-show',
                'time_slot_status': True,
                'duplicate_appt': 1,
            },
            {
                'start_time': day(14),
                'end_time': day(14) + pd.Timedelta(minutes=30),
                'NoShow': False,
                'slot_outcome': 'show',
                'slot_type': 'show',
                'slot_type_detailed': 'show',
                'time_slot_status': False,
                'duplicate_appt': 1,
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df)
        slot_df = build_slot_df(status_df, include_id_cols=True)

        pd.testing.assert_frame_equal(slot_df, expected_slot_df, check_like=True)

    def test_hard_noshow(self):
        raw_df = pd.DataFrame.from_records([
            # date,    now_status,           now_sched_for_date
            (day(0), code['scheduled'], day(7)),
            (day(8), code['scheduled'], day(14)),
            (day(13), code['registered'], day(14)),
            (day(14), code['started'], day(14)),
            (day(14), code['examined'], day(14)),
        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'ambulant'

        expected_slot_df = pd.DataFrame([
            {
                'start_time': day(7),
                'end_time': day(7) + pd.Timedelta(minutes=30),
                'NoShow': True,
                'slot_outcome': 'rescheduled',
                'slot_type': 'no-show',
                'slot_type_detailed': 'hard no-show',
                'time_slot_status': True,
                'duplicate_appt': 1,
            },
            {
                'start_time': day(14),
                'end_time': day(14) + pd.Timedelta(minutes=30),
                'NoShow': False,
                'slot_outcome': 'show',
                'slot_type': 'show',
                'slot_type_detailed': 'show',
                'time_slot_status': False,
                'duplicate_appt': 1,
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df)
        slot_df = build_slot_df(status_df, include_id_cols=True)

        pd.testing.assert_frame_equal(slot_df, expected_slot_df, check_like=True)

    def test_canceled_soft_no_show(self):
        raw_df = pd.DataFrame.from_records([
            # date,    now_status,           now_sched_for_date
            (day(0), code['scheduled'], day(7)),
            (day(6), code['canceled'], day(7)),
        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'ambulant'

        expected_slot_df = pd.DataFrame([
            {
                'start_time': day(7),
                'end_time': day(7) + pd.Timedelta(minutes=30),
                'NoShow': True,
                'slot_outcome': 'canceled',
                'slot_type': 'no-show',
                'slot_type_detailed': 'soft no-show',
                'time_slot_status': True,
                'duplicate_appt': 1,
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df)
        slot_df = build_slot_df(status_df, include_id_cols=True)

        pd.testing.assert_frame_equal(slot_df, expected_slot_df, check_like=True)

    def test_canceled_hard_no_show_not_actually_canceled_status(self):
        raw_df = pd.DataFrame.from_records([
            # date,                             now_status,           now_sched_for_date
            (day(0),                            code['scheduled'], day(7)),
            (day(6),                            code['registered'], day(7)),
            (day(7) + pd.Timedelta(minutes=10), code['scheduled'], day(7)),
        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'ambulant'

        expected_slot_df = pd.DataFrame([
            {
                'start_time': day(7),
                'end_time': day(7) + pd.Timedelta(minutes=30),
                'NoShow': True,
                'slot_outcome': 'canceled',
                'slot_type': 'no-show',
                'slot_type_detailed': 'hard no-show',
                'time_slot_status': True,
                'duplicate_appt': 1,
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df)
        slot_df = build_slot_df(status_df, include_id_cols=True)

        pd.testing.assert_frame_equal(slot_df, expected_slot_df, check_like=True)

    def test_duplicate_appt_show(self):
        return

    # def test_not_a_slot(self):
    #     return True

    # def test_MRNCmpdId_included(self):
    #     return True

    # def test_not_include_id_cols(self):
    #     return True

    # def test_time_of_examined_not_equals_scheduled_plus_30(self):
    #     return True