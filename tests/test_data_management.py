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
                'patient_class_adj': 'ambulant',
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df)
        slot_df = build_slot_df(status_df)

        pd.testing.assert_frame_equal(slot_df, expected_slot_df, check_like=True)

    def test_rescheduled_3_days_in_advance_no_slot(self):
        raw_df = pd.DataFrame.from_records([
                # date,     now_status,            now_sched_for_date
                (day(0),    code['scheduled'],     day(7)),
                (day(4),    code['scheduled'],     day(14)),
            ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'ambulant'

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, None)
        status_df = build_status_df(raw_df)
        slot_df = build_slot_df(status_df)

        self.assertEqual(slot_df.shape[0], 0)

    def test_canceled_three_days_in_advance_not_a_slot(self):
        raw_df = pd.DataFrame.from_records([
            # date,     now_status,            now_sched_for_date
            (day(0),    code['scheduled'],     day(7)),
            (day(4),    code['canceled'],      day(7)),
        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'ambulant'

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, None)
        status_df = build_status_df(raw_df)
        slot_df = build_slot_df(status_df)

        self.assertEqual(slot_df.shape[0], 0)

    def test_inpatient_canceled_one_day_in_advance_not_a_slot(self):
        raw_df = pd.DataFrame.from_records([
            # date,     now_status,            now_sched_for_date
            (day(0),    code['scheduled'],     day(7)),
            (day(6),    code['canceled'],      day(7)),
        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'stationär'

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, None)
        status_df = build_status_df(raw_df)
        slot_df = build_slot_df(status_df)

        self.assertEqual(slot_df.shape[0], 0)

    def test_soft_noshow_rescheduled(self):
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
                'patient_class_adj': 'ambulant',
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
                'patient_class_adj': 'ambulant',
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df)
        slot_df = build_slot_df(status_df)

        pd.testing.assert_frame_equal(slot_df, expected_slot_df, check_like=True)

    def test_hard_noshow_rescheduled(self):
        raw_df = pd.DataFrame.from_records([
            # date,    now_status,           now_sched_for_date
            (day(0),   code['scheduled'],    day(7)),
            (day(8),   code['scheduled'],    day(14)),
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
                'slot_type_detailed': 'hard no-show',
                'time_slot_status': True,
                'duplicate_appt': 1,
                'patient_class_adj': 'ambulant',
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
                'patient_class_adj': 'ambulant',
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df)
        slot_df = build_slot_df(status_df)

        pd.testing.assert_frame_equal(slot_df, expected_slot_df, check_like=True)

    def test_soft_no_show_canceled(self):
        raw_df = pd.DataFrame.from_records([
            # date,    now_status,           now_sched_for_date
            (day(0),   code['scheduled'],    day(7)),
            (day(6),   code['canceled'],     day(7)),
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
                'patient_class_adj': 'ambulant',
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df)
        slot_df = build_slot_df(status_df)

        pd.testing.assert_frame_equal(slot_df, expected_slot_df, check_like=True)

    def test_hard_no_show_canceled(self):
        raw_df = pd.DataFrame.from_records([
            # date,                             now_status,           now_sched_for_date
            (day(0),                            code['scheduled'],    day(7)),
            (day(6),                            code['registered'],   day(7)),
            (day(7) + pd.Timedelta(minutes=10), code['canceled'],     day(7)),
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
                'patient_class_adj': 'ambulant',
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df)
        slot_df = build_slot_df(status_df)

        pd.testing.assert_frame_equal(slot_df, expected_slot_df, check_like=True)

    def test_canceled_hard_no_show_not_actually_canceled_status(self):
        raw_df = pd.DataFrame.from_records([
            # date,                             now_status,           now_sched_for_date
            (day(0),                            code['scheduled'],    day(7)),
            (day(6),                            code['registered'],   day(7)),
            (day(7) + pd.Timedelta(minutes=10), code['scheduled'],    day(7)),
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
                'patient_class_adj': 'ambulant',
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df)
        slot_df = build_slot_df(status_df)

        pd.testing.assert_frame_equal(slot_df, expected_slot_df, check_like=True)

    def test_duplicate_appt_half_canceled_creates_one_show(self):
        raw_df = pd.DataFrame.from_records([
            # fon, date,                               now_status,            now_sched_for_date
            (1,    day(0),                             code['requested'],     day(4)),
            (2,    day(0),                             code['requested'],     day(4)),

            (1,    day(1),                             code['scheduled'],     day(4)),
            (2,    day(1),                             code['scheduled'],     day(4)),

            (1,    day(3),                             code['registered'],    day(4)),
            (2,    day(3),                             code['registered'],    day(4)),
            (2,    day(3) + pd.Timedelta(minutes=30),  code['canceled'],      day(4)),

            (1,    day(4),                             code['started'],       day(4)),
            (1,    day(4) + pd.Timedelta(minutes=30),  code['examined'],      day(4)),


        ],
            columns=['FillerOrderNo', date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'ambulant'

        expected_slot_df = pd.DataFrame([
            {
                'FillerOrderNo': 1,
                'start_time': day(4),
                'end_time': day(4) + pd.Timedelta(minutes=30),
                'NoShow': False,
                'slot_outcome': 'show',
                'slot_type': 'show',
                'slot_type_detailed': 'show',
                'time_slot_status': False,
                'duplicate_appt': 2,
                'patient_class_adj': 'ambulant',
            },
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df, create_fon=False)
        status_df = build_status_df(raw_df)
        slot_df = build_slot_df(status_df)

        pd.testing.assert_frame_equal(slot_df, expected_slot_df, check_like=True)

    def test_duplicate_appt_half_rescheduled_creates_one_show(self):
        raw_df = pd.DataFrame.from_records([
            # fon, date,                               now_status,            now_sched_for_date
            (1,    day(0),                             code['requested'],     day(4)),
            (2,    day(0),                             code['requested'],     day(4)),

            (1,    day(1),                             code['scheduled'],     day(4)),
            (2,    day(1),                             code['scheduled'],     day(4)),

            (1,    day(3),                             code['registered'],    day(4)),
            (2,    day(3),                             code['registered'],    day(4)),
            (2,    day(3) + pd.Timedelta(minutes=30),  code['scheduled'],     day(4)),

            (1,    day(4),                             code['started'],       day(4)),
            (1,    day(4) + pd.Timedelta(minutes=30),  code['examined'],      day(4)),


        ],
            columns=['FillerOrderNo', date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'ambulant'

        expected_slot_df = pd.DataFrame([
            {
                'FillerOrderNo': 1,
                'start_time': day(4),
                'end_time': day(4) + pd.Timedelta(minutes=30),
                'NoShow': False,
                'slot_outcome': 'show',
                'slot_type': 'show',
                'slot_type_detailed': 'show',
                'time_slot_status': False,
                'duplicate_appt': 2,
                'patient_class_adj': 'ambulant',
            },
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df, create_fon=False)
        status_df = build_status_df(raw_df)
        slot_df = build_slot_df(status_df)

        pd.testing.assert_frame_equal(slot_df, expected_slot_df, check_like=True)

    def test_duplicate_appt_both_canceled_creates_one_no_show(self):
        raw_df = pd.DataFrame.from_records([
            # fon, date,                               now_status,            now_sched_for_date
            (1,    day(0),                             code['requested'],     day(4)),
            (2,    day(0),                             code['requested'],     day(4)),

            (1,    day(1),                             code['scheduled'],     day(4)),
            (2,    day(1),                             code['scheduled'],     day(4)),

            (1,    day(3),                             code['registered'],    day(4)),
            (2,    day(3),                             code['registered'],    day(4)),

            (1,    day(4) + pd.Timedelta(minutes=10),  code['canceled'],      day(4)),
            (2,    day(4) + pd.Timedelta(minutes=10),  code['canceled'],      day(4)),


        ],
            columns=['FillerOrderNo', date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'ambulant'

        expected_slot_df = pd.DataFrame([
            {
                'FillerOrderNo': 1,
                'start_time': day(4),
                'end_time': day(4) + pd.Timedelta(minutes=30),
                'NoShow': True,
                'slot_outcome': 'canceled',
                'slot_type': 'no-show',
                'slot_type_detailed': 'hard no-show',
                'time_slot_status': True,
                'duplicate_appt': 2,
                'patient_class_adj': 'ambulant',
            },
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df, create_fon=False)
        status_df = build_status_df(raw_df)
        slot_df = build_slot_df(status_df)

        pd.testing.assert_frame_equal(slot_df, expected_slot_df, check_like=True)

    def test_not_include_id_cols(self):
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
                'patient_class_adj': 'ambulant',
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        expected_slot_df.drop(columns=['FillerOrderNo', 'MRNCmpdId'], inplace=True)
        status_df = build_status_df(raw_df)
        slot_df = build_slot_df(status_df, include_id_cols=False)

        pd.testing.assert_frame_equal(slot_df, expected_slot_df, check_like=True)

    def test_slot_end_time_not_same_as_status_change_timestamp_to_examined(self):
        raw_df = pd.DataFrame.from_records([
            # date,                               now_status,            now_sched_for_date
            (day(0),                              code['scheduled'],     day(4)),
            (day(4),                              code['started'],       day(4)),
            (day(4) + pd.Timedelta(minutes=27),   code['examined'],      day(4)),
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
                'patient_class_adj': 'ambulant',
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df)
        slot_df = build_slot_df(status_df)

        pd.testing.assert_frame_equal(slot_df, expected_slot_df, check_like=True)

