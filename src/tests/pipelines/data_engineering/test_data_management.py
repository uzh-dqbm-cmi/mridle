import unittest
import pandas as pd
import numpy as np
# from mridle.pipelines.data_engineering.dispo import build_dispo_df, find_no_shows_from_dispo_exp_two


def day(num_days_from_start, hour=9):
    return pd.Timestamp(year=2019, month=1, day=1, hour=hour, minute=0) + pd.Timedelta(days=num_days_from_start)

# TODO: add back when create dispo pipeline, rename this file
# class TestExperimentTwoDataProcessing(unittest.TestCase):
#
#     def test_show(self):
#         dispo_exp_2_records = [
#             {
#                 'patient_id': 1,
#                 'date': str(day(3).date()),
#                 'start_time': str(day(3).time()),
#                 'machine': 'MR1',
#                 'type': 'ter',
#                 'date_recorded': str(day(0).date()),
#             },
#             {
#                 'patient_id': 1,
#                 'date': str(day(3).date()),
#                 'start_time': str(day(3).time()),
#                 'machine': 'MR1',
#                 'type': 'bef',
#                 'date_recorded': str(day(6).date()),  # skip over weekend
#             },
#
#         ]
#
#         expected = pd.DataFrame([
#             {
#                 'patient_id': 1,
#                 'date': day(3, hour=0),
#                 'start_time': day(3),
#                 'machine_before': 'MR1',
#                 'type_before': 'ter',
#                 'date_recorded_before': day(0, hour=0),
#                 'date_diff_before': -3,
#                 'machine_after': 'MR1',
#                 'type_after': 'bef',
#                 'date_recorded_after': day(6, hour=0),
#                 'date_diff_after': 1,
#                 'NoShow': None,
#                 'slot_outcome': None,
#             }
#         ])
#         dispo_exp_2_df = build_dispo_df(dispo_exp_2_records, test_patient_ids=[])
#         result = find_no_shows_from_dispo_exp_two(dispo_exp_2_df)
#         pd.testing.assert_frame_equal(result, expected, check_like=True)
#
#     def test_no_show(self):
#         dispo_exp_2_records = [
#             {
#                 'patient_id': 1,
#                 'date': str(day(2).date()),
#                 'start_time': str(day(2).time()),
#                 'machine': 'MR1',
#                 'type': 'ter',
#                 'date_recorded': str(day(0).date()),
#             },
#
#         ]
#
#         expected = pd.DataFrame({
#             'patient_id': pd.Series([1]),
#             'date': pd.Series([day(2, hour=0)]),
#             'start_time': pd.Series([day(2)]),
#             'machine_before': pd.Series(['MR1']),
#             'type_before': pd.Series(['ter']),
#             'date_recorded_before': pd.Series([day(0, hour=0)]),
#             'date_diff_before': pd.Series([-2]),
#             'machine_after': pd.Series([np.NaN], dtype=str),
#             'type_after': pd.Series([np.NaN], dtype=str),
#             'date_recorded_after': pd.Series([np.datetime64('NaT')]),
#             'date_diff_after': pd.Series([np.NaN]),
#             'NoShow': pd.Series([True]),
#             'slot_outcome': pd.Series(['rescheduled']),
#         })
#         dispo_exp_2_df = build_dispo_df(dispo_exp_2_records, test_patient_ids=[])
#         result = find_no_shows_from_dispo_exp_two(dispo_exp_2_df)
#         pd.testing.assert_frame_equal(result, expected, check_like=True)
#
#     def test_no_slot_for_three_days_advance(self):
#         dispo_exp_2_records = [
#             {
#                 'patient_id': 1,
#                 'date': str(day(3).date()),
#                 'start_time': str(day(3).time()),
#                 'machine': 'MR1',
#                 'type': 'ter',
#                 'date_recorded': str(day(0).date()),
#             },
#
#         ]
#
#         expected = pd.DataFrame({
#             'patient_id': pd.Series([1]),
#             'date': pd.Series([day(3, hour=0)]),
#             'start_time': pd.Series([day(3)]),
#             'machine_before': pd.Series(['MR1']),
#             'type_before': pd.Series(['ter']),
#             'date_recorded_before': pd.Series([day(0, hour=0)]),
#             'date_diff_before': pd.Series([-3]),
#             'machine_after': pd.Series([np.NaN], dtype=str),
#             'type_after': pd.Series([np.NaN], dtype=str),
#             'date_recorded_after': pd.Series([np.datetime64('NaT')]),
#             'date_diff_after': pd.Series([np.NaN]),
#             'NoShow': pd.Series([None]),
#             'slot_outcome': pd.Series([None]),
#         })
#         dispo_exp_2_df = build_dispo_df(dispo_exp_2_records, test_patient_ids=[])
#         result = find_no_shows_from_dispo_exp_two(dispo_exp_2_df)
#         pd.testing.assert_frame_equal(result, expected, check_like=True)
#
#     def test_change_in_time_within_same_day_yields_resched_and_show(self):
#         dispo_exp_2_records = [
#             {
#                 'patient_id': 1,
#                 'date': str(day(1).date()),
#                 'start_time': str(day(1, hour=14).time()),
#                 'machine': 'MR1',
#                 'type': 'ter',
#                 'date_recorded': str(day(0).date()),
#             },
#             {
#                 'patient_id': 1,
#                 'date': str(day(1).date()),
#                 'start_time': str(day(1, hour=13).time()),  # time change!
#                 'machine': 'MR1',
#                 'type': 'bef',
#                 'date_recorded': str(day(2).date()),
#             },
#
#         ]
#
#         expected = pd.DataFrame([
#             {
#                 'patient_id': 1,
#                 'date': day(1, hour=0),
#                 'start_time': day(1, hour=14),
#                 'machine_before': 'MR1',
#                 'type_before': 'ter',
#                 'date_recorded_before': day(0, hour=0),
#                 'date_diff_before': -1,
#                 'machine_after': None,
#                 'type_after': None,
#                 'date_recorded_after': None,
#                 'date_diff_after': None,
#                 'NoShow': True,
#                 'slot_outcome': 'rescheduled',
#             },
#             {
#                 'patient_id': 1,
#                 'date': day(1, hour=0),
#                 'start_time': day(1, hour=13),
#                 'machine_before': None,
#                 'type_before': None,
#                 'date_recorded_before': None,
#                 'date_diff_before': None,
#                 'machine_after': 'MR1',
#                 'type_after': 'bef',
#                 'date_recorded_after': day(2, hour=0),
#                 'date_diff_after': 1,
#                 'NoShow': False,
#                 'slot_outcome': 'show',
#             }
#         ])
#         dispo_exp_2_df = build_dispo_df(dispo_exp_2_records, test_patient_ids=[])
#         result = find_no_shows_from_dispo_exp_two(dispo_exp_2_df)
#         pd.testing.assert_frame_equal(result, expected, check_like=True)
#
#     def test_midnight_show(self):
#         dispo_exp_2_records = [
#             {
#                 'patient_id': 1,
#                 'date': str(day(1).date()),
#                 'start_time': str(day(1, hour=0).time()),
#                 'machine': 'MR1',
#                 'type': 'ter',
#                 'date_recorded': str(day(0).date()),
#             },
#             {
#                 'patient_id': 1,
#                 'date': str(day(1).date()),
#                 'start_time': str(day(1, hour=0).time()),  # time change!
#                 'machine': 'MR1',
#                 'type': 'bef',
#                 'date_recorded': str(day(2).date()),
#             },
#
#         ]
#
#         expected = pd.DataFrame([
#             {
#                 'patient_id': 1,
#                 'date': day(1, hour=0),
#                 'start_time': day(1, hour=0),
#                 'machine_before': 'MR1',
#                 'type_before': 'ter',
#                 'date_recorded_before': day(0, hour=0),
#                 'date_diff_before': -1,
#                 'machine_after': 'MR1',
#                 'type_after': 'bef',
#                 'date_recorded_after': day(2, hour=0),
#                 'date_diff_after': 1,
#                 'NoShow': False,
#                 'slot_outcome': 'show',
#             }
#         ])
#         dispo_exp_2_df = build_dispo_df(dispo_exp_2_records, test_patient_ids=[])
#         result = find_no_shows_from_dispo_exp_two(dispo_exp_2_df)
#         pd.testing.assert_frame_equal(result, expected, check_like=True)
#
#     def test_midnight_reschedule_yields_no_slot_assumed_inpatient(self):
#         dispo_exp_2_records = [
#             {
#                 'patient_id': 1,
#                 'date': str(day(1).date()),
#                 'start_time': str(day(1, hour=0).time()),
#                 'machine': 'MR1',
#                 'type': 'ter',
#                 'date_recorded': str(day(0).date()),
#             },
#         ]
#
#         expected = pd.DataFrame([
#             {
#                 'patient_id': 1,
#                 'date': day(1, hour=0),
#                 'start_time': day(1, hour=0),
#                 'machine_before': 'MR1',
#                 'type_before': 'ter',
#                 'date_recorded_before': day(0, hour=0),
#                 'date_diff_before': -1,
#                 'machine_after': None,
#                 'type_after': None,
#                 'date_recorded_after': np.datetime64('NaT'),
#                 'date_diff_after': np.NaN,
#                 'NoShow': False,
#                 'slot_outcome': None,
#             }
#         ])
#
#         dispo_exp_2_df = build_dispo_df(dispo_exp_2_records, test_patient_ids=[])
#         result = find_no_shows_from_dispo_exp_two(dispo_exp_2_df)
#         pd.testing.assert_frame_equal(result, expected, check_like=True)
