import unittest
import pandas as pd
import numpy as np
from mridle.pipelines.data_science.feature_engineering.nodes import feature_no_show_before
from mridle.pipelines.data_engineering.ris.nodes import build_status_df, build_slot_df, find_no_shows, \
    set_no_show_severity, STATUS_MAP
from mridle.pipelines.data_science.feature_engineering.nodes import build_feature_set, generate_training_data, \
    generate_3_5_days_ahead_features, feature_days_scheduled_in_advance, generate_3_5_days_ahead_features


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
                'FillerOrderNo': 1,
                'start_time': str(day(2)),
                'NoShow': 1
            },
            {
                'MRNCmpdId': 1,
                'FillerOrderNo': 1,
                'start_time': str(day(3)),
                'NoShow': 1
            },
            {
                'MRNCmpdId': 1,
                'FillerOrderNo': 1,
                'start_time': str(day(4)),
                'NoShow': 0
            }
        ])

        expected = pd.DataFrame([
            {
                'MRNCmpdId': 1,
                'FillerOrderNo': 1,
                'start_time': str(day(2)),
                'NoShow': 1,
                'no_show_before': 0,
                'no_show_before_sq': 0,
                'appts_before': 0,
                'show_before': 0,
                'no_show_rate': 0.0
            },
            {
                'MRNCmpdId': 1,
                'FillerOrderNo': 1,
                'start_time': str(day(3)),
                'NoShow': 1,
                'no_show_before': 1,
                'no_show_before_sq': 1,
                'appts_before': 1,
                'show_before': 0,
                'no_show_rate': 1.0

            },
            {
                'MRNCmpdId': 1,
                'FillerOrderNo': 1,
                'start_time': str(day(4)),
                'NoShow': 0,
                'no_show_before': 2,
                'no_show_before_sq': 4,
                'appts_before': 2,
                'show_before': 0,
                'no_show_rate': 1.0

            }
        ])
        result = feature_no_show_before(slot_df)

        pd.testing.assert_frame_equal(result, expected, check_like=True)

    def test_no_show_before_no_no_shows(self):
        slot_df = pd.DataFrame([
            {
                'MRNCmpdId': 2,
                'FillerOrderNo': 1,
                'start_time': str(day(2)),
                'NoShow': 0
            },
            {
                'MRNCmpdId': 2,
                'FillerOrderNo': 1,
                'start_time': str(day(3)),
                'NoShow': 0
            }
        ])

        expected = pd.DataFrame([
            {
                'MRNCmpdId': 2,
                'FillerOrderNo': 1,
                'start_time': str(day(2)),
                'NoShow': 0,
                'no_show_before': 0,
                'no_show_before_sq': 0,
                'appts_before': 0,
                'show_before': 0,
                'no_show_rate': 0.0
            },
            {
                'MRNCmpdId': 2,
                'FillerOrderNo': 1,
                'start_time': str(day(3)),
                'NoShow': 0,
                'no_show_before': 0,
                'no_show_before_sq': 0,
                'appts_before': 1,
                'show_before': 1,
                'no_show_rate': 0.0
            }
        ])
        result = feature_no_show_before(slot_df)

        pd.testing.assert_frame_equal(result, expected, check_like=True)

    def test_no_show_before_two_patients(self):
        slot_df = pd.DataFrame([
            {
                'MRNCmpdId': 1,
                'FillerOrderNo': 1,
                'start_time': str(day(2)),
                'NoShow': 1
            },
            {
                'MRNCmpdId': 1,
                'FillerOrderNo': 1,
                'start_time': str(day(3)),
                'NoShow': 1
            },
            {
                'MRNCmpdId': 1,
                'FillerOrderNo': 1,
                'start_time': str(day(4)),
                'NoShow': 0
            },
            {
                'MRNCmpdId': 2,
                'FillerOrderNo': 2,
                'start_time': str(day(2)),
                'NoShow': 0
            },
            {
                'MRNCmpdId': 2,
                'FillerOrderNo': 2,
                'start_time': str(day(3)),
                'NoShow': 0
            }
        ])

        expected = pd.DataFrame([
            {
                'MRNCmpdId': 1,
                'FillerOrderNo': 1,
                'start_time': str(day(2)),
                'NoShow': 1,
                'no_show_before': 0,
                'no_show_before_sq': 0,
                'appts_before': 0,
                'show_before': 0,
                'no_show_rate': 0.0

            },
            {
                'MRNCmpdId': 1,
                'FillerOrderNo': 1,
                'start_time': str(day(3)),
                'NoShow': 1,
                'no_show_before': 1,
                'no_show_before_sq': 1,
                'appts_before': 1,
                'show_before': 0,
                'no_show_rate': 1.0

            },
            {
                'MRNCmpdId': 1,
                'FillerOrderNo': 1,
                'start_time': str(day(4)),
                'NoShow': 0,
                'no_show_before': 2,
                'no_show_before_sq': 4,
                'appts_before': 2,
                'show_before': 0,
                'no_show_rate': 1.0

            },
            {
                'MRNCmpdId': 2,
                'FillerOrderNo': 2,
                'start_time': str(day(2)),
                'NoShow': 0,
                'no_show_before': 0,
                'no_show_before_sq': 0,
                'appts_before': 0,
                'show_before': 0,
                'no_show_rate': 0.0

            },
            {
                'MRNCmpdId': 2,
                'FillerOrderNo': 2,
                'start_time': str(day(3)),
                'NoShow': 0,
                'no_show_before': 0,
                'no_show_before_sq': 0,
                'appts_before': 1,
                'show_before': 1,
                'no_show_rate': 0.0

            }
        ])
        result = feature_no_show_before(slot_df)

        pd.testing.assert_frame_equal(result, expected, check_like=True)

    def test_no_show_before_correct_ordering(self):
        slot_df = pd.DataFrame([
            {
                'MRNCmpdId': 2,
                'FillerOrderNo': 1,
                'start_time': str(day(3)),
                'NoShow': 0
            },
            {
                'MRNCmpdId': 2,
                'FillerOrderNo': 1,
                'start_time': str(day(2)),
                'NoShow': 1
            }
        ])

        expected = pd.DataFrame([
            {
                'MRNCmpdId': 2,
                'FillerOrderNo': 1,
                'start_time': str(day(3)),
                'NoShow': 0,
                'no_show_before': 1,
                'no_show_before_sq': 1,
                'appts_before': 1,
                'show_before': 0,
                'no_show_rate': 1.0

            },
            {
                'MRNCmpdId': 2,
                'FillerOrderNo': 1,
                'start_time': str(day(2)),
                'NoShow': 1,
                'no_show_before': 0,
                'no_show_before_sq': 0,
                'appts_before': 0,
                'show_before': 0,
                'no_show_rate': 0.0
            }
        ])
        result = feature_no_show_before(slot_df)
        print(result.columns)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True), check_like=True)


class TestDaysScheduleInAdvance(unittest.TestCase):

    @staticmethod
    def _fill_out_static_columns(raw_df, slot_df=None, create_fon=True):
        if create_fon:
            raw_df['FillerOrderNo'] = 0
        raw_df['MRNCmpdId'] = '0'
        raw_df['EnteringOrganisationDeviceID'] = 'MR1'
        raw_df['UniversalServiceName'] = 'MR'
        raw_df['OrderStatus'] = raw_df[now_status_col].tail(1).iloc[0]
        raw_df['ReasonForStudy'] = 'cancer'

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

        return raw_df

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
        slot_df = build_slot_df(status_df, valid_date_range, build_future_slots=True)
        past_slot_df = build_slot_df(status_df, valid_date_range, build_future_slots=False)
        slot_df_with_feature = feature_days_scheduled_in_advance(status_df, slot_df)
        slot_df_with_feature = slot_df_with_feature.merge(past_slot_df[['MRNCmpdId', 'FillerOrderNo', 'start_time']],
                                                          how='inner', on=['MRNCmpdId', 'FillerOrderNo', 'start_time'])
        pd.testing.assert_frame_equal(slot_df_with_feature, expected_slot_df, check_like=True)


class TestFutureSlots(unittest.TestCase):
    @staticmethod
    def _fill_out_static_columns(raw_df, slot_df=None, create_fon=True):
        if create_fon:
            raw_df['FillerOrderNo'] = 0
        raw_df['MRNCmpdId'] = '0'
        raw_df['EnteringOrganisationDeviceID'] = 'MR1'
        raw_df['UniversalServiceName'] = 'MR'
        raw_df['OrderStatus'] = raw_df[now_status_col].tail(1).iloc[0]
        raw_df['Klasse'] = 'A'
        raw_df['Sex'] = 'weiblich'
        raw_df['DateOfBirth'] = '05-03-1994'
        raw_df['Zivilstand'] = 'UNB'
        raw_df['Zip'] = '8001'
        raw_df['Beruf'] = 'Arzt'
        raw_df['ReasonForStudy'] = 'cancer'
        
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
                                 'UniversalServiceName'
                                 ]
            slot_df = slot_df[slot_df_col_order]
            return raw_df, slot_df

        return raw_df

    def test_future_appointments_one_row(self):
        raw_df = pd.DataFrame.from_records([
            # date,                               now_status,            now_sched_for_date
            (day(0), code['scheduled'], day(7)),
        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'ambulant'

        expected_feature_df = pd.DataFrame([
            {
                'start_time': day(7),
                'end_time': day(7) + pd.Timedelta(minutes=30),
                'NoShow': False,
                'sched_days_advanced': 7
            }
        ])

        raw_df = self._fill_out_static_columns(raw_df, create_fon=True)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        status_df['Telefon'] = '0'
        feature_df = build_feature_set(status_df, valid_date_range)
        cols = [c for c in expected_feature_df.columns.values]
        feature_df = feature_df.loc[:, feature_df.columns.isin(cols)]
        pd.testing.assert_frame_equal(feature_df, expected_feature_df, check_like=True)

    def test_future_appointments_moved_forward(self):
        raw_df = pd.DataFrame.from_records([
            # date,                               now_status,            now_sched_for_date
            (day(0), code['requested'], day(20)),
            (day(1), code['scheduled'], day(8)),
        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'ambulant'

        expected_feature_df = pd.DataFrame([
            {
                'start_time': day(8),
                'end_time': day(8) + pd.Timedelta(minutes=30),
                'NoShow': False,
                'sched_days_advanced': 7
            }
        ])

        raw_df = self._fill_out_static_columns(raw_df, create_fon=True)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        status_df['Telefon'] = '0'

        test_dt = day(2)
        feature_df = generate_3_5_days_ahead_features(status_df, test_dt)

        cols = [c for c in expected_feature_df.columns.values]
        feature_df = feature_df.loc[:, feature_df.columns.isin(cols)]
        feature_df = feature_df.reindex(cols, axis=1)
        pd.testing.assert_frame_equal(feature_df, expected_feature_df, check_like=True)

    def test_future_appointments_rescheduled(self):
        raw_df = pd.DataFrame.from_records([
            # date,                               now_status,            now_sched_for_date
            (day(0), code['requested'], day(13)),
            (day(1), code['scheduled'], day(6)),
            (day(7), code['scheduled'], day(16)),
        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'ambulant'

        expected_feature_df = pd.DataFrame([
            {
                'start_time': day(16),
                'end_time': day(16) + pd.Timedelta(minutes=30),
                'NoShow': False,
                'sched_days_advanced': 9
            }
        ])

        raw_df = self._fill_out_static_columns(raw_df, create_fon=True)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        status_df['Telefon'] = '0'
        feature_df = generate_training_data(status_df, valid_date_range)

        cols = [c for c in expected_feature_df.columns.values]
        feature_df = feature_df.loc[:, feature_df.columns.isin(cols)]
        feature_df = feature_df.reindex(cols, axis=1)
        pd.testing.assert_frame_equal(feature_df, expected_feature_df, check_like=True)

    def test_multiple_appts(self):
        raw_df_1 = pd.DataFrame.from_records([
            # date,                               now_status,            now_sched_for_date
            (day(0), code['requested'], day(6)),
            (day(1), code['scheduled'], day(9)),
            (day(9), code['started'], day(9)),
            (day(9) + pd.Timedelta(minutes=30), code['examined'], day(7)),

        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df_1['PatientClass'] = 'ambulant'

        raw_df_2 = pd.DataFrame.from_records([
            # date,                               now_status,            now_sched_for_date
            (day(0), code['requested'], day(13)),
            (day(1), code['scheduled'], day(8)),
            (day(9), code['scheduled'], day(16)),
        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df_2['PatientClass'] = 'ambulant'

        raw_df_3 = pd.DataFrame.from_records([
            # date,                               now_status,            now_sched_for_date
            (day(0), code['requested'], day(13)),
            (day(1), code['scheduled'], day(9)),
            (day(8), code['scheduled'], day(13)),
        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df_3['PatientClass'] = 'ambulant'

        raw_df_4 = pd.DataFrame.from_records([  # this appt should show up as upcoming, but not as noshow when resched.
            # date,                               now_status,            now_sched_for_date
            (day(0), code['requested'], day(13)),
            (day(1), code['scheduled'], day(9)),
            (day(3), code['scheduled'], day(13)),
        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df_4['PatientClass'] = 'ambulant'

        expected_model_data_df = pd.DataFrame([  # All NoShow will be False, since we aren't merging NoShow info on
            {
                'MRNCmpdId': '0',
                'FillerOrderNo': 0,
                'start_time': day(6),
                'end_time': day(6) + pd.Timedelta(minutes=30),
                'NoShow': False
            },
            {
                'MRNCmpdId': '0',
                'FillerOrderNo': 0,
                'start_time': day(9),
                'end_time': day(9) + pd.Timedelta(minutes=30),
                'NoShow': False
            },
            {
                'MRNCmpdId': '1',
                'FillerOrderNo': 1,
                'start_time': day(8),
                'end_time': day(8) + pd.Timedelta(minutes=30),
                'NoShow': False
            },
            {
                'MRNCmpdId': '2',
                'FillerOrderNo': 2,
                'start_time': day(9),
                'end_time': day(9) + pd.Timedelta(minutes=30),
                'NoShow': False
            },
            {
                'MRNCmpdId': '3',
                'FillerOrderNo': 3,
                'start_time': day(9),
                'end_time': day(9) + pd.Timedelta(minutes=30),
                'NoShow': False
            }
        ])

        raw_df_1 = self._fill_out_static_columns(raw_df_1, create_fon=True)
        raw_df_2 = self._fill_out_static_columns(raw_df_2, create_fon=True)
        raw_df_3 = self._fill_out_static_columns(raw_df_3, create_fon=True)
        raw_df_4 = self._fill_out_static_columns(raw_df_4, create_fon=True)
        raw_df_2['MRNCmpdId'] = '1'
        raw_df_2['FillerOrderNo'] = 1
        raw_df_3['MRNCmpdId'] = '2'
        raw_df_3['FillerOrderNo'] = 2
        raw_df_4['MRNCmpdId'] = '3'
        raw_df_4['FillerOrderNo'] = 3
        raw_df = pd.concat([raw_df_1, raw_df_2, raw_df_3, raw_df_4], axis=0)

        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        status_df['Telefon'] = '0'

        # Set valid date range to as if we were generating this data on day(12)
        test_vdr = [pd.Timestamp(year=2019, month=1, day=1, hour=0, minute=0),
                    pd.Timestamp(year=2019, month=1, day=13, hour=0, minute=0)]
        model_data_df = generate_training_data(status_df, test_vdr, append_outcome=False)
        cols = [c for c in expected_model_data_df.columns.values]
        model_data_df = model_data_df.loc[:, model_data_df.columns.isin(cols)]
        model_data_df = model_data_df.reindex(cols, axis=1)
        model_data_df = model_data_df.sort_values('MRNCmpdId').reset_index(drop=True)

        pd.testing.assert_frame_equal(model_data_df, expected_model_data_df, check_like=True)



class TestGenerateModelData(unittest.TestCase):
    @staticmethod
    def _fill_out_static_columns(raw_df, slot_df=None, create_fon=True):
        if create_fon:
            raw_df['FillerOrderNo'] = 0
        raw_df['MRNCmpdId'] = '0'
        raw_df['EnteringOrganisationDeviceID'] = 'MR1'
        raw_df['UniversalServiceName'] = 'MR'
        raw_df['OrderStatus'] = raw_df[now_status_col].tail(1).iloc[0]
        raw_df['Klasse'] = 'A'
        raw_df['Sex'] = 'weiblich'
        raw_df['DateOfBirth'] = '05-03-1994'
        raw_df['Zivilstand'] = 'UNB'
        raw_df['Zip'] = '8001'
        raw_df['Beruf'] = 'Arzt'
        raw_df['ReasonForStudy'] = 'cancer'

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
                                 'UniversalServiceName'
                                 ]
            slot_df = slot_df[slot_df_col_order]
            return raw_df, slot_df

        return raw_df

    def test_appointment_one_row(self):
        raw_df = pd.DataFrame.from_records([
            # date,                               now_status,            now_sched_for_date
            (day(0), code['scheduled'], day(7)),
            (day(7), code['started'], day(7)),
            (day(7) + pd.Timedelta(minutes=30), code['examined'], day(7)),
        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df['PatientClass'] = 'ambulant'

        expected_model_data_df = pd.DataFrame([
            {
                'start_time': day(7),
                'end_time': day(7) + pd.Timedelta(minutes=30),
                'NoShow': False,
                'slot_outcome': 'show',
                'slot_type': 'show',
                'slot_type_detailed': 'show'
            }
        ])

        raw_df = self._fill_out_static_columns(raw_df, create_fon=True)
        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        status_df['Telefon'] = '0'

        model_data_df = generate_training_data(status_df, valid_date_range)

        cols = [c for c in expected_model_data_df.columns.values]
        model_data_df = model_data_df.loc[:, model_data_df.columns.isin(cols)]
        model_data_df = model_data_df.reindex(cols, axis=1)
        pd.testing.assert_frame_equal(model_data_df, expected_model_data_df, check_like=True)

    def test_multiple_appts(self):
        raw_df_1 = pd.DataFrame.from_records([
            # date,                               now_status,            now_sched_for_date
            (day(0), code['requested'], day(5)),
            (day(1), code['scheduled'], day(7)),
            (day(7), code['started'], day(7)),
            (day(7) + pd.Timedelta(minutes=30), code['examined'], day(7)),

        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df_1['PatientClass'] = 'ambulant'

        raw_df_2 = pd.DataFrame.from_records([
            # date,                               now_status,            now_sched_for_date
            (day(0), code['requested'], day(13)),
            (day(1), code['scheduled'], day(8)),
            (day(9), code['scheduled'], day(16)),
        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df_2['PatientClass'] = 'ambulant'

        raw_df_3 = pd.DataFrame.from_records([
            # date,                               now_status,            now_sched_for_date
            (day(0), code['requested'], day(13)),
            (day(1), code['scheduled'], day(9)),
            (day(8), code['scheduled'], day(13)),
        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df_3['PatientClass'] = 'ambulant'

        raw_df_4 = pd.DataFrame.from_records([  # this appt should show up as upcoming, but not as noshow when resched.
            # date,                               now_status,            now_sched_for_date
            (day(0), code['requested'], day(13)),
            (day(1), code['scheduled'], day(9)),
            (day(3), code['scheduled'], day(13)),
        ],
            columns=[date_col, now_status_col, now_sched_for_date_col]
        )
        raw_df_4['PatientClass'] = 'ambulant'

        expected_model_data_df = pd.DataFrame([
            {
                'MRNCmpdId': '0',
                'FillerOrderNo': 0,
                'start_time': day(7),
                'end_time': day(7) + pd.Timedelta(minutes=30),
                'NoShow': False,
                'slot_outcome': 'show',
                'slot_type': 'show',
                'slot_type_detailed': 'show'
            },
            {
                'MRNCmpdId': '1',
                'FillerOrderNo': 1,
                'start_time': day(8),
                'end_time': day(8) + pd.Timedelta(minutes=30),
                'NoShow': True,
                'slot_outcome': 'rescheduled',
                'slot_type': 'no-show',
                'slot_type_detailed': 'hard no-show'
            },
            {
                'MRNCmpdId': '2',
                'FillerOrderNo': 2,
                'start_time': day(9),
                'end_time': day(9) + pd.Timedelta(minutes=30),
                'NoShow': True,
                'slot_outcome': 'rescheduled',
                'slot_type': 'no-show',
                'slot_type_detailed': 'soft no-show'
            },
            {
                'MRNCmpdId': '3',
                'FillerOrderNo': 3,
                'start_time': day(9),
                'end_time': day(9) + pd.Timedelta(minutes=30),
                'NoShow': False,
                'slot_outcome': 'ok_rescheduled',
                'slot_type': 'ok_rescheduled',
                'slot_type_detailed': 'ok_rescheduled'
            }
        ])

        raw_df_1 = self._fill_out_static_columns(raw_df_1, create_fon=True)
        raw_df_2 = self._fill_out_static_columns(raw_df_2, create_fon=True)
        raw_df_3 = self._fill_out_static_columns(raw_df_3, create_fon=True)
        raw_df_4 = self._fill_out_static_columns(raw_df_4, create_fon=True)
        raw_df_2['MRNCmpdId'] = '1'
        raw_df_2['FillerOrderNo'] = 1
        raw_df_3['MRNCmpdId'] = '2'
        raw_df_3['FillerOrderNo'] = 2
        raw_df_4['MRNCmpdId'] = '3'
        raw_df_4['FillerOrderNo'] = 3
        raw_df = pd.concat([raw_df_1, raw_df_2, raw_df_3, raw_df_4], axis=0)

        status_df = build_status_df(raw_df, exclude_patient_ids=[])
        status_df['Telefon'] = '0'

        # Set valid date range to as if we were generating this data on day(12)
        test_vdr = [pd.Timestamp(year=2019, month=1, day=1, hour=0, minute=0),
                    pd.Timestamp(year=2019, month=1, day=13, hour=0, minute=0)]
        model_data_df = generate_training_data(status_df, test_vdr, append_outcome=True)
        cols = [c for c in expected_model_data_df.columns.values]
        model_data_df = model_data_df.loc[:, model_data_df.columns.isin(cols)]
        model_data_df = model_data_df.reindex(cols, axis=1)
        model_data_df = model_data_df.sort_values('MRNCmpdId').reset_index(drop=True)

        pd.testing.assert_frame_equal(model_data_df, expected_model_data_df, check_like=True)
