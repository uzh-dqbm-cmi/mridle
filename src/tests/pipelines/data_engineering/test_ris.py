import unittest
import pandas as pd
import numpy as np
from mridle.pipelines.data_engineering.ris.nodes import build_status_df, build_slot_df, find_no_shows,\
    set_no_show_severity, STATUS_MAP


code = {status: letter_code for letter_code, status in STATUS_MAP.items()}


def day(num_days_from_start, hour=9):
    """Helper function for concisely creating dates"""
    return pd.Timestamp(year=2019, month=1, day=1, hour=hour, minute=0) + pd.Timedelta(days=num_days_from_start)


valid_date_range = [pd.Timestamp(year=2019, month=1, day=1, hour=0, minute=0),
                    pd.Timestamp(year=2019, month=2, day=1, hour=0, minute=0)]

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
                'duplicate_appt': 1,
                'patient_class_adj': 'ambulant',
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, valid_date_range)
        print(expected_slot_df.T, slot_df.T)

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
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, valid_date_range)

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
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, valid_date_range)

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
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, valid_date_range)

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
                'duplicate_appt': 1,
                'patient_class_adj': 'ambulant',
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, valid_date_range)

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
                'duplicate_appt': 1,
                'patient_class_adj': 'ambulant',
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, valid_date_range)

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
                'duplicate_appt': 1,
                'patient_class_adj': 'ambulant',
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, valid_date_range)

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
                'duplicate_appt': 1,
                'patient_class_adj': 'ambulant',
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, valid_date_range)

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
                'duplicate_appt': 1,
                'patient_class_adj': 'ambulant',
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, valid_date_range)

        pd.testing.assert_frame_equal(slot_df, expected_slot_df, check_like=True)

    def test_appt_false_start_and_start_time_moved_ahead(self):
        raw_df = pd.DataFrame.from_records([
            # date,                             now_status,           now_sched_for_date
            (day(0),                            code['scheduled'],    day(5)),
            (day(4),                            code['registered'],   day(5)),
            (day(5),                            code['started'],      day(5)),
            (day(5) + pd.Timedelta(minutes=10), code['scheduled'],    day(8)),
            (day(7),                            code['registered'],   day(8)),
            (day(8),                            code['started'],      day(8)),
            (day(8) + pd.Timedelta(minutes=10), code['examined'],     day(8)),
        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'stationär'

        expected_slot_df = pd.DataFrame([
            {
                'start_time': day(8),
                'end_time': day(8) + pd.Timedelta(minutes=30),
                'NoShow': False,
                'slot_outcome': 'show',
                'slot_type': 'inpatient',
                'slot_type_detailed': 'inpatient',
                'duplicate_appt': 0,
                'patient_class_adj': 'inpatient',
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, valid_date_range)

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
                'duplicate_appt': 2,
                'patient_class_adj': 'ambulant',
            },
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df, create_fon=False)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, valid_date_range)

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
                'duplicate_appt': 2,
                'patient_class_adj': 'ambulant',
            },
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df, create_fon=False)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, valid_date_range)

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
                'duplicate_appt': 2,
                'patient_class_adj': 'ambulant',
            },
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df, create_fon=False)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, valid_date_range)

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
                'duplicate_appt': 1,
                'patient_class_adj': 'ambulant',
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        expected_slot_df.drop(columns=['FillerOrderNo', 'MRNCmpdId'], inplace=True)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, valid_date_range, include_id_cols=False)

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
                'duplicate_appt': 1,
                'patient_class_adj': 'ambulant',
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, valid_date_range)

        pd.testing.assert_frame_equal(slot_df, expected_slot_df, check_like=True)

    def test_basic_show_not_included_when_outside_valid_date_range(self):
        raw_df = pd.DataFrame.from_records([
                # date,                               now_status,            now_sched_for_date
                (day(0),                              code['scheduled'],     day(4)),
                (day(4),                              code['started'],       day(4)),
                (day(4) + pd.Timedelta(minutes=30),   code['examined'],      day(4)),
            ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'ambulant'

        different_valid_date_range = [pd.Timestamp(year=2020, month=1, day=1, hour=0, minute=0),
                                      pd.Timestamp(year=2020, month=2, day=1, hour=0, minute=0)]

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, None)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, different_valid_date_range)

        self.assertEqual(len(slot_df), 0)

    def test_future_appointments_one_row(self):
        raw_df = pd.DataFrame.from_records([
                # date,                               now_status,            now_sched_for_date
                (day(0),                              code['scheduled'],     day(4)),
            ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'ambulant'

        expected_slot_df = pd.DataFrame([
            {
                'start_time': day(4),
                'end_time': day(4) + pd.Timedelta(minutes=30),
                'NoShow': False,
                'slot_outcome': None,
                'slot_type': None,
                'slot_type_detailed': None,
                'duplicate_appt': 1,
                'patient_class_adj': 'ambulant',
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, valid_date_range, build_future_slots=True)

        pd.testing.assert_frame_equal(slot_df, expected_slot_df, check_like=True)

    def test_future_appointments_multiple_rows(self):
        raw_df = pd.DataFrame.from_records([
                # date,                               now_status,            now_sched_for_date
                (day(0),                              code['requested'],     day(3)),
                (day(1),                              code['scheduled'],     day(4)),
            ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'ambulant'

        expected_slot_df = pd.DataFrame([
            {
                'start_time': day(4),
                'end_time': day(4) + pd.Timedelta(minutes=30),
                'NoShow': False,
                'slot_outcome': None,
                'slot_type': None,
                'slot_type_detailed': None,
                'duplicate_appt': 1,
                'patient_class_adj': 'ambulant',
            }
        ])

        raw_df, expected_slot_df = self._fill_out_static_columns(raw_df, expected_slot_df)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        slot_df = build_slot_df(status_df, valid_date_range, build_future_slots=True)

        pd.testing.assert_frame_equal(slot_df, expected_slot_df, check_like=True)


class TestFindNoShowsPositive(unittest.TestCase):

    def test_rescheduled_on_same_day(self):
        example_row = pd.DataFrame({
            'patient_class_adj': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'scheduled',
            'now_status': 'scheduled',
            'first_created_date': pd.Timestamp(year=2018, month=12, day=1, hour=9),
        }, index=[0])
        self.assertTrue(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_rescheduled_one_day_prior(self):
        example_row = pd.DataFrame({
            'patient_class_adj': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=2, hour=9),
            'was_status': 'scheduled',
            'now_status': 'scheduled',
            'first_created_date': pd.Timestamp(year=2018, month=12, day=1, hour=9),
        }, index=[0])
        self.assertTrue(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_registered_to_scheduled_on_same_day(self):
        example_row = pd.DataFrame({
            'patient_class_adj': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'registered',
            'now_status': 'scheduled',
            'first_created_date': pd.Timestamp(year=2018, month=12, day=1, hour=9),
        }, index=[0])
        self.assertTrue(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_registered_to_canceled_on_same_day(self):
        example_row = pd.DataFrame({
            'patient_class_adj': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'registered',
            'now_status': 'canceled',
            'first_created_date': pd.Timestamp(year=2018, month=12, day=1, hour=9),
        }, index=[0])
        self.assertTrue(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_scheduled_to_canceled_one_day_prior(self):
        example_row = pd.DataFrame({
            'patient_class_adj': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=2, hour=9),
            'was_status': 'scheduled',
            'now_status': 'canceled',
            'first_created_date': pd.Timestamp(year=2018, month=12, day=1, hour=9),
        }, index=[0])
        self.assertTrue(example_row.apply(find_no_shows, axis=1).iloc[0])


class TestFindNoShowsNegative(unittest.TestCase):

    def test_rescheduled_three_days_in_advance(self):
        example_row = pd.DataFrame({
            'patient_class_adj': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=4, hour=9),
            'was_status': 'scheduled',
            'now_status': 'scheduled',
            'first_created_date': pd.Timestamp(year=2018, month=12, day=1, hour=9),
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_rescheduled_registered_to_scheduled_three_days_in_advance(self):
        example_row = pd.DataFrame({
            'patient_class_adj': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=4, hour=9),
            'was_status': 'registered',
            'now_status': 'scheduled',
            'first_created_date': pd.Timestamp(year=2018, month=12, day=1, hour=9),
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_inpatient_rescheduled_same_day(self):
        example_row = pd.DataFrame({
            'patient_class_adj': 'inpatient',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'scheduled',
            'now_status': 'scheduled',
            'first_created_date': pd.Timestamp(year=2018, month=12, day=1, hour=9),
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_changed_from_requested_to_scheduled_same_day(self):
        example_row = pd.DataFrame({
            'patient_class_adj': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'requested',
            'now_status': 'scheduled',
            'first_created_date': pd.Timestamp(year=2018, month=12, day=1, hour=9),
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_changed_from_requested_to_requested_same_day(self):
        example_row = pd.DataFrame({
            'patient_class_adj': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'requested',
            'now_status': 'requested',
            'first_created_date': pd.Timestamp(year=2018, month=12, day=1, hour=9),
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_changed_from_scheduled_to_registered_same_day(self):
        example_row = pd.DataFrame({
            'patient_class_adj': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'scheduled',
            'now_status': 'registered',
            'first_created_date': pd.Timestamp(year=2018, month=12, day=1, hour=9),
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_changed_from_registered_to_waiting_same_day(self):
        example_row = pd.DataFrame({
            'patient_class_adj': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'registered',
            'now_status': 'waiting',
            'first_created_date': pd.Timestamp(year=2018, month=12, day=1, hour=9),
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_changed_from_waiting_to_started_same_day(self):
        example_row = pd.DataFrame({
            'patient_class_adj': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'waiting',
            'now_status': 'started',
            'first_created_date': pd.Timestamp(year=2018, month=12, day=1, hour=9),
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_changed_from_started_to_examined_same_day(self):
        example_row = pd.DataFrame({
            'patient_class_adj': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'started',
            'now_status': 'examined',
            'first_created_date': pd.Timestamp(year=2018, month=12, day=1, hour=9),
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_changed_from_scheduled_to_written_same_day(self):
        example_row = pd.DataFrame({
            'patient_class_adj': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'scheduled',
            'now_status': 'written',
            'first_created_date': pd.Timestamp(year=2018, month=12, day=1, hour=9),
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_no_was_status(self):
        example_row = pd.DataFrame({
            'patient_class_adj': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': None,
            'now_status': 'scheduled',
            'first_created_date': pd.Timestamp(year=2018, month=12, day=1, hour=9),
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_no_was_sched_for_date(self):
        example_row = pd.DataFrame({
            'patient_class_adj': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': None,
            'was_status': 'scheduled',
            'now_status': 'scheduled',
            'first_created_date': pd.Timestamp(year=2018, month=12, day=1, hour=9),
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_was_sched_for_date_is_midnight(self):
        example_row = pd.DataFrame({
            'patient_class_adj': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=2, hour=0),
            'was_status': 'scheduled',
            'now_status': 'scheduled',
            'first_created_date': pd.Timestamp(year=2018, month=12, day=1, hour=9),
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_first_created_date_is_same_as_reschedule_date(self):
        example_row = pd.DataFrame({
            'patient_class_adj': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=10),
            'was_status': 'scheduled',
            'now_status': 'scheduled',
            'first_created_date': pd.Timestamp(year=2019, month=1, day=1, hour=8),
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])


class TestSetNoShowSeverity(unittest.TestCase):

    def test_not_no_show_blank(self):
        example_row = pd.DataFrame({
            'NoShow': False,
        }, index=[0])
        expected_result = None
        self.assertEqual(example_row.apply(set_no_show_severity, axis=1).iloc[0], expected_result)

    def test_rescheduled_two_days_in_advance_soft(self):
        example_row = pd.DataFrame({
            'date': pd.Timestamp(year=2019, month=1, day=2),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=4, hour=9),
            'NoShow': True,
        }, index=[0])
        expected_result = 'soft'
        self.assertEqual(example_row.apply(set_no_show_severity, axis=1).iloc[0], expected_result)

    def test_rescheduled_one_day_in_advance_soft(self):
        example_row = pd.DataFrame({
            'date': pd.Timestamp(year=2019, month=1, day=3),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=4, hour=9),
            'NoShow': True,
        }, index=[0])
        expected_result = 'soft'
        self.assertEqual(example_row.apply(set_no_show_severity, axis=1).iloc[0], expected_result)

    def test_rescheduled_one_hour_later_hard(self):
        example_row = pd.DataFrame({
            'date': pd.Timestamp(year=2019, month=1, day=4, hour=10),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=4, hour=9),
            'NoShow': True,
        }, index=[0])
        expected_result = 'hard'
        self.assertEqual(example_row.apply(set_no_show_severity, axis=1).iloc[0], expected_result)

    def test_rescheduled_one_day_later_hard(self):
        example_row = pd.DataFrame({
            'date': pd.Timestamp(year=2019, month=1, day=5, hour=9),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=4, hour=9),
            'NoShow': True,
        }, index=[0])
        expected_result = 'hard'
        self.assertEqual(example_row.apply(set_no_show_severity, axis=1).iloc[0], expected_result)

