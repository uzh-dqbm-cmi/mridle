import unittest
import pandas as pd
import numpy as np
from mridle.data_management import aggregate_terminplanner, build_slot_df, build_dispo_df, find_no_shows_from_dispo_exp_two,\
    STATUS_MAP
from mridle.idle_time import calc_idle_time_gaps, calc_daily_idle_time_stats, calc_appts_and_gaps

code = {status: letter_code for letter_code, status in STATUS_MAP.items()}


def day(num_days_from_start, hour=9):
    return pd.Timestamp(year=2019, month=1, day=1, hour=hour, minute=0) + pd.Timedelta(days=num_days_from_start)


class TestTerminplannerAggregation(unittest.TestCase):

    def test_one_appt(self):

        terminplanner_dummy = pd.DataFrame([
            {
                'Terminbuch': 'MR1',
                'Wochentag': 'MO',
                'TERMINRASTER_NAME': 'not_required',
                'gültig von': pd.to_datetime(day(-10), format='%d.%m.%Y'),
                'gültig bis': pd.to_datetime(day(10), format='%d.%m.%Y'),
                'Termin': '14:00',
                'Dauer in Min.': 120
            }
        ])

        dicom_data_dummy = pd.DataFrame([
            {
                'AccessionNumber': 1,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=30),
                'image_end': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=5)
            }
        ])

        expected_total_day_time = 2.0
        expected_active_time = 35
        expected_buffer_time = 10
        expected_idle_time = 75

        expected = pd.DataFrame([{
            'image_device_id': 1,
            'total_day_time': expected_total_day_time,  # hours
            'active': expected_active_time/60,  # fraction of hours
            'idle': expected_idle_time/60,  # fraction of hours
            'buffer': expected_buffer_time/60,  # fraction of hours
            'active_pct': (expected_active_time/60) / expected_total_day_time,
            'idle_pct': (expected_idle_time/60) / expected_total_day_time,
            'buffer_pct': (expected_buffer_time/60) / expected_total_day_time
        }])

        tp_agg = aggregate_terminplanner(terminplanner_dummy)
        idle_df = calc_idle_time_gaps(dicom_data_dummy, tp_agg, time_buffer_mins=5)

        # stats = calc_daily_idle_time_stats(idle_df)
        # result = stats[['total_day_time', 'active_time', 'idle_time', 'buffer_time', 'active_time_pct',
        #                 'idle_time_pct', 'buffer_time_pct']]
        appts_and_gaps = calc_appts_and_gaps(idle_df)

        appts_and_gaps['status_duration'] = (appts_and_gaps['end'] - appts_and_gaps['start']) / pd.to_timedelta(1,
                                                                                                                unit='H')

        daily_stats = appts_and_gaps.groupby(['date', 'image_device_id', 'status']).agg({
            'status_duration': 'sum'
        }).reset_index()
        one_hour = pd.to_timedelta(1, unit='H')
        total_day_times = appts_and_gaps.groupby(['date', 'image_device_id']).agg({
            'start': 'min',
            'end': 'max'
        }).reset_index()
        total_day_times['total_day_time'] = (total_day_times['end'] - total_day_times['start']) / one_hour
        stats = daily_stats.pivot_table(columns='status', values='status_duration',
                                        index=['date', 'image_device_id']).reset_index()
        stats = stats.merge(total_day_times, on=['date', 'image_device_id'])
        stats['active'] = stats['total_day_time'] - stats['buffer'] - stats[
            'idle']  # as there might be overlaps in appts, so we don't want to doublecount
        stats['active_pct'] = stats['active'] / stats['total_day_time']
        stats['buffer_pct'] = stats['buffer'] / stats['total_day_time']
        stats['idle_pct'] = stats['idle'] / stats['total_day_time']

        result = stats[
            ['image_device_id', 'total_day_time', 'active', 'idle', 'buffer', 'active_pct', 'idle_pct', 'buffer_pct']]
        pd.testing.assert_frame_equal(result, expected, check_like=True)


    def test_multiple_appts(self):
        terminplanner_dummy = pd.DataFrame([
            {
                'Terminbuch': 'MR1',
                'Wochentag': 'MO',
                'TERMINRASTER_NAME': 'not_required',
                'gültig von': pd.to_datetime(day(-10), format='%d.%m.%Y'),
                'gültig bis': pd.to_datetime(day(10), format='%d.%m.%Y'),
                'Termin': '14:00',
                'Dauer in Min.': 120
            }
        ])
        dicom_data_dummy = pd.DataFrame([
            {
                'AccessionNumber': 1,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=30),
                'image_end': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=5)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=20),
                'image_end': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=50)
            }
        ])

        expected_total_day_time = 2.0
        expected_active_time = 35 + 30
        expected_buffer_time = 10 + 10
        expected_idle_time = 35

        expected = pd.DataFrame([{
            'total_day_time': expected_total_day_time,  # hours
            'active': expected_active_time / 60,  # fraction of hours
            'idle': expected_idle_time / 60,  # fraction of hours
            'buffer': expected_buffer_time / 60,  # fraction of hours
            'active_pct': (expected_active_time / 60) / expected_total_day_time,
            'idle_pct': (expected_idle_time / 60) / expected_total_day_time,
            'buffer_pct': (expected_buffer_time / 60) / expected_total_day_time
        }])

        tp_agg = aggregate_terminplanner(terminplanner_dummy)
        idle_df = calc_idle_time_gaps(dicom_data_dummy, tp_agg, time_buffer_mins=5)

        # stats = calc_daily_idle_time_stats(idle_df)
        # result = stats[['total_day_time', 'active_time', 'idle_time', 'buffer_time', 'active_time_pct',
        #                 'idle_time_pct', 'buffer_time_pct']]
        appts_and_gaps = calc_appts_and_gaps(idle_df)

        appts_and_gaps['status_duration'] = (appts_and_gaps['end'] - appts_and_gaps['start']) / pd.to_timedelta(1,
                                                                                                                unit='H')
        stats = appts_and_gaps.groupby('status').agg({'status_duration': 'sum'})
        total_day_time = (stats.loc['active'] + stats.loc['idle'] + stats.loc['buffer'])[0].astype(float)
        result = pd.DataFrame([{
            'total_day_time': total_day_time,  # hours
            'active': stats.loc['active'][0],  # fraction of hours
            'idle': stats.loc['idle'][0],  # fraction of hours
            'buffer': stats.loc['buffer'][0],  # fraction of hours
            'active_pct': (stats.loc['active'][0]) / total_day_time,
            'idle_pct': (stats.loc['idle'][0]) / total_day_time,
            'buffer_pct': (stats.loc['buffer'][0]) / total_day_time
        }])

        pd.testing.assert_frame_equal(result, expected, check_like=True)


    def test_two_machines(self):
        terminplanner_dummy = pd.DataFrame([
            {
                'Terminbuch': 'MR1',
                'Wochentag': 'MO',
                'TERMINRASTER_NAME': 'not_required',
                'gültig von': pd.to_datetime(day(-10), format='%d.%m.%Y'),
                'gültig bis': pd.to_datetime(day(10), format='%d.%m.%Y'),
                'Termin': '14:00',
                'Dauer in Min.': 120
            },
            {
                'Terminbuch': 'MR2',
                'Wochentag': 'MO',
                'TERMINRASTER_NAME': 'not_required',
                'gültig von': pd.to_datetime(day(-10), format='%d.%m.%Y'),
                'gültig bis': pd.to_datetime(day(10), format='%d.%m.%Y'),
                'Termin': '14:00',
                'Dauer in Min.': 60
            }
        ])
        dicom_data_dummy = pd.DataFrame([
            {
                'AccessionNumber': 1,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=30),
                'image_end': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=5)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=20),
                'image_end': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=50)
            },
            {
                'AccessionNumber': 3,
                'StudyDescription': '-',
                'image_device_id': 2,
                'image_start': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=20),
                'image_end': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=45)
            }
        ])

        expected_total_day_time1 = 2.0
        expected_active_time1 = 35 + 30
        expected_buffer_time1 = 10 + 10
        expected_idle_time1 = 25 + 5 + 5

        expected_total_day_time2 = 1.0
        expected_active_time2 = 25
        expected_buffer_time2 = 10
        expected_idle_time2 = 25

        expected = pd.DataFrame([{
            'image_device_id': 1,
            'total_day_time': expected_total_day_time1,  # hours
            'active': expected_active_time1 / 60,  # fraction of hours
            'idle': expected_idle_time1 / 60,  # fraction of hours
            'buffer': expected_buffer_time1 / 60,  # fraction of hours
            'active_pct': (expected_active_time1 / 60) / expected_total_day_time1,
            'idle_pct': (expected_idle_time1 / 60) / expected_total_day_time1,
            'buffer_pct': (expected_buffer_time1 / 60) / expected_total_day_time1
        },
        {
            'image_device_id': 2,
            'total_day_time': expected_total_day_time2,  # hours
            'active': expected_active_time2 / 60,  # fraction of hours
            'idle': expected_idle_time2 / 60,  # fraction of hours
            'buffer': expected_buffer_time2 / 60,  # fraction of hours
            'active_pct': (expected_active_time2 / 60) / expected_total_day_time2,
            'idle_pct': (expected_idle_time2 / 60) / expected_total_day_time2,
            'buffer_pct': (expected_buffer_time2 / 60) / expected_total_day_time2
        }
        ])

        tp_agg = aggregate_terminplanner(terminplanner_dummy)
        idle_df = calc_idle_time_gaps(dicom_data_dummy, tp_agg, time_buffer_mins=5)

        # stats = calc_daily_idle_time_stats(idle_df)
        # result = stats[['total_day_time', 'active_time', 'idle_time', 'buffer_time', 'active_time_pct',
        #                 'idle_time_pct', 'buffer_time_pct']]
        appts_and_gaps = calc_appts_and_gaps(idle_df)

        appts_and_gaps['status_duration'] = (appts_and_gaps['end'] - appts_and_gaps['start']) / pd.to_timedelta(1,
                                                                                                                unit='H')

        daily_stats = appts_and_gaps.groupby(['date', 'image_device_id', 'status']).agg({
            'status_duration': 'sum'
        }).reset_index()
        one_hour = pd.to_timedelta(1, unit='H')
        total_day_times = appts_and_gaps.groupby(['date', 'image_device_id']).agg({
            'start': 'min',
            'end': 'max'
        }).reset_index()
        total_day_times['total_day_time'] = (total_day_times['end'] - total_day_times['start'])/one_hour
        stats = daily_stats.pivot_table(columns='status', values='status_duration', index=['date', 'image_device_id']).reset_index()
        stats = stats.merge(total_day_times, on=['date', 'image_device_id'])
        stats['active'] = stats['total_day_time'] - stats['buffer'] - stats['idle']  # as there might be overlaps in appts, so we don't want to doublecount
        stats['active_pct'] = stats['active'] / stats['total_day_time']
        stats['buffer_pct'] = stats['buffer'] / stats['total_day_time']
        stats['idle_pct'] = stats['idle'] / stats['total_day_time']

        result = stats[['image_device_id', 'total_day_time', 'active', 'idle', 'buffer','active_pct', 'idle_pct', 'buffer_pct']]
        pd.testing.assert_frame_equal(result, expected, check_like=True)

    def test_overlapping_buffer(self):
        terminplanner_dummy = pd.DataFrame([
            {
                'Terminbuch': 'MR1',
                'Wochentag': 'MO',
                'TERMINRASTER_NAME': 'not_required',
                'gültig von': pd.to_datetime(day(-10), format='%d.%m.%Y'),
                'gültig bis': pd.to_datetime(day(10), format='%d.%m.%Y'),
                'Termin': '14:00',
                'Dauer in Min.': 120
            }
        ])
        dicom_data_dummy = pd.DataFrame([
            {
                'AccessionNumber': 1,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=20),
                'image_end': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=50)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=55),
                'image_end': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=15)
            },
            {
                'AccessionNumber': 3,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=20),
                'image_end': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=50)
            }
        ])

        expected_total_day_time1 = 2.0
        expected_active_time1 = 80
        expected_buffer_time1 = 20
        expected_idle_time1 = 20

        expected = pd.DataFrame([{
            'image_device_id': 1,
            'total_day_time': expected_total_day_time1,  # hours
            'active': expected_active_time1 / 60,  # fraction of hours
            'idle': expected_idle_time1 / 60,  # fraction of hours
            'buffer': expected_buffer_time1 / 60,  # fraction of hours
            'active_pct': (expected_active_time1 / 60) / expected_total_day_time1,
            'idle_pct': (expected_idle_time1 / 60) / expected_total_day_time1,
            'buffer_pct': (expected_buffer_time1 / 60) / expected_total_day_time1
        }
        ])

        tp_agg = aggregate_terminplanner(terminplanner_dummy)
        idle_df = calc_idle_time_gaps(dicom_data_dummy, tp_agg, time_buffer_mins=5)

        # stats = calc_daily_idle_time_stats(idle_df)
        # result = stats[['total_day_time', 'active_time', 'idle_time', 'buffer_time', 'active_time_pct',
        #                 'idle_time_pct', 'buffer_time_pct']]
        appts_and_gaps = calc_appts_and_gaps(idle_df)

        appts_and_gaps['status_duration'] = (appts_and_gaps['end'] - appts_and_gaps['start']) / pd.to_timedelta(1,
                                                                                                                unit='H')

        daily_stats = appts_and_gaps.groupby(['date', 'image_device_id', 'status']).agg({
            'status_duration': 'sum'
        }).reset_index()
        one_hour = pd.to_timedelta(1, unit='H')
        total_day_times = appts_and_gaps.groupby(['date', 'image_device_id']).agg({
            'start': 'min',
            'end': 'max'
        }).reset_index()
        total_day_times['total_day_time'] = (total_day_times['end'] - total_day_times['start']) / one_hour
        stats = daily_stats.pivot_table(columns='status', values='status_duration',
                                        index=['date', 'image_device_id']).reset_index()
        stats = stats.merge(total_day_times, on=['date', 'image_device_id'])
        stats['active'] = stats['total_day_time'] - stats['buffer'] - stats[
            'idle']  # as there might be overlaps in appts, so we don't want to doublecount
        stats['active_pct'] = stats['active'] / stats['total_day_time']
        stats['buffer_pct'] = stats['buffer'] / stats['total_day_time']
        stats['idle_pct'] = stats['idle'] / stats['total_day_time']

        result = stats[
            ['image_device_id', 'total_day_time', 'active', 'idle', 'buffer', 'active_pct', 'idle_pct', 'buffer_pct']]
        pd.testing.assert_frame_equal(result, expected, check_like=True)


    def test_overlapping_appts(self):
        terminplanner_dummy = pd.DataFrame([
            {
                'Terminbuch': 'MR1',
                'Wochentag': 'MO',
                'TERMINRASTER_NAME': 'not_required',
                'gültig von': pd.to_datetime(day(-10), format='%d.%m.%Y'),
                'gültig bis': pd.to_datetime(day(10), format='%d.%m.%Y'),
                'Termin': '14:00',
                'Dauer in Min.': 120
            }
        ])
        dicom_data_dummy = pd.DataFrame([
            {
                'AccessionNumber': 1,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=20),
                'image_end': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=50)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=45),
                'image_end': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=15)
            }
        ])

        expected_total_day_time1 = 2.0
        expected_active_time1 = 55
        expected_buffer_time1 = 10
        expected_idle_time1 = 55

        expected = pd.DataFrame([{
            'image_device_id': 1,
            'total_day_time': expected_total_day_time1,  # hours
            'active': expected_active_time1 / 60,  # fraction of hours
            'idle': expected_idle_time1 / 60,  # fraction of hours
            'buffer': expected_buffer_time1 / 60,  # fraction of hours
            'active_pct': (expected_active_time1 / 60) / expected_total_day_time1,
            'idle_pct': (expected_idle_time1 / 60) / expected_total_day_time1,
            'buffer_pct': (expected_buffer_time1 / 60) / expected_total_day_time1
        }
        ])

        tp_agg = aggregate_terminplanner(terminplanner_dummy)
        idle_df = calc_idle_time_gaps(dicom_data_dummy, tp_agg, time_buffer_mins=5)

        # stats = calc_daily_idle_time_stats(idle_df)
        # result = stats[['total_day_time', 'active_time', 'idle_time', 'buffer_time', 'active_time_pct',
        #                 'idle_time_pct', 'buffer_time_pct']]
        appts_and_gaps = calc_appts_and_gaps(idle_df)

        appts_and_gaps['status_duration'] = (appts_and_gaps['end'] - appts_and_gaps['start']) / pd.to_timedelta(1,
                                                                                                                unit='H')

        daily_stats = appts_and_gaps.groupby(['date', 'image_device_id', 'status']).agg({
            'status_duration': 'sum'
        }).reset_index()
        one_hour = pd.to_timedelta(1, unit='H')
        total_day_times = appts_and_gaps.groupby(['date', 'image_device_id']).agg({
            'start': 'min',
            'end': 'max'
        }).reset_index()
        total_day_times['total_day_time'] = (total_day_times['end'] - total_day_times['start']) / one_hour
        stats = daily_stats.pivot_table(columns='status', values='status_duration',
                                        index=['date', 'image_device_id']).reset_index()
        stats = stats.merge(total_day_times, on=['date', 'image_device_id'])
        stats['active'] = stats['total_day_time'] - stats['buffer'] - stats[
            'idle']  # as there might be overlaps in appts, so we don't want to doublecount
        stats['active_pct'] = stats['active'] / stats['total_day_time']
        stats['buffer_pct'] = stats['buffer'] / stats['total_day_time']
        stats['idle_pct'] = stats['idle'] / stats['total_day_time']

        result = stats[
            ['image_device_id', 'total_day_time', 'active', 'idle', 'buffer', 'active_pct', 'idle_pct', 'buffer_pct']]
        pd.testing.assert_frame_equal(result, expected, check_like=True)


    def test_appt_starting_before_day(self):
        terminplanner_dummy = pd.DataFrame([
            {
                'Terminbuch': 'MR1',
                'Wochentag': 'MO',
                'TERMINRASTER_NAME': 'not_required',
                'gültig von': pd.to_datetime(day(-10), format='%d.%m.%Y'),
                'gültig bis': pd.to_datetime(day(10), format='%d.%m.%Y'),
                'Termin': '14:00',
                'Dauer in Min.': 120
            }
        ])
        dicom_data_dummy = pd.DataFrame([
            {
                'AccessionNumber': 1,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=13) + pd.Timedelta(minutes=55),
                'image_end': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=20)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=45),
                'image_end': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=15)
            }
        ])

        expected_total_day_time1 = 2.0 + 5/60
        expected_active_time1 = 55
        expected_buffer_time1 = 15
        expected_idle_time1 = 55

        expected = pd.DataFrame([{
            'image_device_id': 1,
            'total_day_time': expected_total_day_time1,  # hours
            'active': expected_active_time1 / 60,  # fraction of hours
            'idle': expected_idle_time1 / 60,  # fraction of hours
            'buffer': expected_buffer_time1 / 60,  # fraction of hours
            'active_pct': (expected_active_time1 / 60) / expected_total_day_time1,
            'idle_pct': (expected_idle_time1 / 60) / expected_total_day_time1,
            'buffer_pct': (expected_buffer_time1 / 60) / expected_total_day_time1
        }
        ])

        tp_agg = aggregate_terminplanner(terminplanner_dummy)

        idle_df = calc_idle_time_gaps(dicom_data_dummy, tp_agg, time_buffer_mins=5)
        # stats = calc_daily_idle_time_stats(idle_df)
        # result = stats[['total_day_time', 'active_time', 'idle_time', 'buffer_time', 'active_time_pct',
        #                 'idle_time_pct', 'buffer_time_pct']]
        appts_and_gaps = calc_appts_and_gaps(idle_df)

        appts_and_gaps['status_duration'] = (appts_and_gaps['end'] - appts_and_gaps['start']) / pd.to_timedelta(1,
                                                                                                                unit='H')

        daily_stats = appts_and_gaps.groupby(['date', 'image_device_id', 'status']).agg({
            'status_duration': 'sum'
        }).reset_index()
        one_hour = pd.to_timedelta(1, unit='H')
        total_day_times = appts_and_gaps.groupby(['date', 'image_device_id']).agg({
            'start': 'min',
            'end': 'max'
        }).reset_index()
        total_day_times['total_day_time'] = (total_day_times['end'] - total_day_times['start']) / one_hour
        stats = daily_stats.pivot_table(columns='status', values='status_duration',
                                        index=['date', 'image_device_id']).reset_index()
        stats = stats.merge(total_day_times, on=['date', 'image_device_id'])
        stats['active'] = stats['total_day_time'] - stats['buffer'] - stats[
            'idle']  # as there might be overlaps in appts, so we don't want to doublecount
        stats['active_pct'] = stats['active'] / stats['total_day_time']
        stats['buffer_pct'] = stats['buffer'] / stats['total_day_time']
        stats['idle_pct'] = stats['idle'] / stats['total_day_time']

        result = stats[
            ['image_device_id', 'total_day_time', 'active', 'idle', 'buffer', 'active_pct', 'idle_pct', 'buffer_pct']]
        pd.testing.assert_frame_equal(result, expected, check_like=True)


    def test_appt_ending_after_day(self):
        terminplanner_dummy = pd.DataFrame([
            {
                'Terminbuch': 'MR1',
                'Wochentag': 'MO',
                'TERMINRASTER_NAME': 'not_required',
                'gültig von': pd.to_datetime(day(-10), format='%d.%m.%Y'),
                'gültig bis': pd.to_datetime(day(10), format='%d.%m.%Y'),
                'Termin': '14:00',
                'Dauer in Min.': 120
            }
        ])
        dicom_data_dummy = pd.DataFrame([
            {
                'AccessionNumber': 1,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=20),
                'image_end': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=55)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=35),
                'image_end': day(num_days_from_start=6, hour=16) + pd.Timedelta(minutes=10)
            }
        ])

        expected_total_day_time1 = 2.0 + 10/60
        expected_active_time1 = 70
        expected_buffer_time1 = 15
        expected_idle_time1 = 45

        expected = pd.DataFrame([{
            'image_device_id': 1,
            'total_day_time': expected_total_day_time1,  # hours
            'active': expected_active_time1 / 60,  # fraction of hours
            'idle': expected_idle_time1 / 60,  # fraction of hours
            'buffer': expected_buffer_time1 / 60,  # fraction of hours
            'active_pct': (expected_active_time1 / 60) / expected_total_day_time1,
            'idle_pct': (expected_idle_time1 / 60) / expected_total_day_time1,
            'buffer_pct': (expected_buffer_time1 / 60) / expected_total_day_time1
        }
        ])

        tp_agg = aggregate_terminplanner(terminplanner_dummy)

        idle_df = calc_idle_time_gaps(dicom_data_dummy, tp_agg, time_buffer_mins=5)
        # stats = calc_daily_idle_time_stats(idle_df)
        # result = stats[['total_day_time', 'active_time', 'idle_time', 'buffer_time', 'active_time_pct',
        #                 'idle_time_pct', 'buffer_time_pct']]
        appts_and_gaps = calc_appts_and_gaps(idle_df)

        appts_and_gaps['status_duration'] = (appts_and_gaps['end'] - appts_and_gaps['start']) / pd.to_timedelta(1,
                                                                                                                unit='H')

        daily_stats = appts_and_gaps.groupby(['date', 'image_device_id', 'status']).agg({
            'status_duration': 'sum'
        }).reset_index()
        one_hour = pd.to_timedelta(1, unit='H')
        total_day_times = appts_and_gaps.groupby(['date', 'image_device_id']).agg({
            'start': 'min',
            'end': 'max'
        }).reset_index()
        total_day_times['total_day_time'] = (total_day_times['end'] - total_day_times['start']) / one_hour
        stats = daily_stats.pivot_table(columns='status', values='status_duration',
                                        index=['date', 'image_device_id']).reset_index()
        stats = stats.merge(total_day_times, on=['date', 'image_device_id'])
        stats['active'] = stats['total_day_time'] - stats['buffer'] - stats[
            'idle']  # as there might be overlaps in appts, so we don't want to doublecount
        stats['active_pct'] = stats['active'] / stats['total_day_time']
        stats['buffer_pct'] = stats['buffer'] / stats['total_day_time']
        stats['idle_pct'] = stats['idle'] / stats['total_day_time']

        result = stats[
            ['image_device_id', 'total_day_time', 'active', 'idle', 'buffer', 'active_pct', 'idle_pct', 'buffer_pct']]
        pd.testing.assert_frame_equal(result, expected, check_like=True)

    def test_two_days(self):
        terminplanner_dummy = pd.DataFrame([
            {
                'Terminbuch': 'MR1',
                'Wochentag': 'MO',
                'TERMINRASTER_NAME': 'not_required',
                'gültig von': pd.to_datetime(day(-10), format='%d.%m.%Y'),
                'gültig bis': pd.to_datetime(day(10), format='%d.%m.%Y'),
                'Termin': '14:00',
                'Dauer in Min.': 120
            },
            {
                'Terminbuch': 'MR1',
                'Wochentag': 'DI',
                'TERMINRASTER_NAME': 'not_required',
                'gültig von': pd.to_datetime(day(-10), format='%d.%m.%Y'),
                'gültig bis': pd.to_datetime(day(10), format='%d.%m.%Y'),
                'Termin': '14:00',
                'Dauer in Min.': 60
            }
        ])
        dicom_data_dummy = pd.DataFrame([
            {
                'AccessionNumber': 1,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=30),
                'image_end': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=5)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=20),
                'image_end': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=50)
            },
            {
                'AccessionNumber': 3,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=7, hour=14) + pd.Timedelta(minutes=20),
                'image_end': day(num_days_from_start=7, hour=14) + pd.Timedelta(minutes=45)
            }
        ])

        expected_total_day_time1 = 2.0
        expected_active_time1 = 35 + 30
        expected_buffer_time1 = 10 + 10
        expected_idle_time1 = 25 + 5 + 5

        expected_total_day_time2 = 1.0
        expected_active_time2 = 25
        expected_buffer_time2 = 10
        expected_idle_time2 = 25

        expected = pd.DataFrame([{
            'date': day(num_days_from_start=6, hour=0),
            'image_device_id': 1,
            'total_day_time': expected_total_day_time1,  # hours
            'active': expected_active_time1 / 60,  # fraction of hours
            'idle': expected_idle_time1 / 60,  # fraction of hours
            'buffer': expected_buffer_time1 / 60,  # fraction of hours
            'active_pct': (expected_active_time1 / 60) / expected_total_day_time1,
            'idle_pct': (expected_idle_time1 / 60) / expected_total_day_time1,
            'buffer_pct': (expected_buffer_time1 / 60) / expected_total_day_time1
        },
        {
            'date': day(num_days_from_start=7, hour=0),
            'image_device_id': 1,
            'total_day_time': expected_total_day_time2,  # hours
            'active': expected_active_time2 / 60,  # fraction of hours
            'idle': expected_idle_time2 / 60,  # fraction of hours
            'buffer': expected_buffer_time2 / 60,  # fraction of hours
            'active_pct': (expected_active_time2 / 60) / expected_total_day_time2,
            'idle_pct': (expected_idle_time2 / 60) / expected_total_day_time2,
            'buffer_pct': (expected_buffer_time2 / 60) / expected_total_day_time2
        }
        ])

        tp_agg = aggregate_terminplanner(terminplanner_dummy)
        idle_df = calc_idle_time_gaps(dicom_data_dummy, tp_agg, time_buffer_mins=5)

        # stats = calc_daily_idle_time_stats(idle_df)
        # result = stats[['total_day_time', 'active_time', 'idle_time', 'buffer_time', 'active_time_pct',
        #                 'idle_time_pct', 'buffer_time_pct']]
        appts_and_gaps = calc_appts_and_gaps(idle_df)

        appts_and_gaps['status_duration'] = (appts_and_gaps['end'] - appts_and_gaps['start']) / pd.to_timedelta(1, unit='H')

        daily_stats = appts_and_gaps.groupby(['date', 'image_device_id', 'status']).agg({
            'status_duration': 'sum'
        }).reset_index()
        one_hour = pd.to_timedelta(1, unit='H')
        total_day_times = appts_and_gaps.groupby(['date', 'image_device_id']).agg({
            'start': 'min',
            'end': 'max'
        }).reset_index()
        total_day_times['total_day_time'] = (total_day_times['end'] - total_day_times['start'])/one_hour
        stats = daily_stats.pivot_table(columns='status', values='status_duration', index=['date', 'image_device_id']).reset_index()
        stats = stats.merge(total_day_times, on=['date', 'image_device_id'])
        stats['active'] = stats['total_day_time'] - stats['buffer'] - stats['idle']  # as there might be overlaps in appts, so we don't want to doublecount
        stats['active_pct'] = stats['active'] / stats['total_day_time']
        stats['buffer_pct'] = stats['buffer'] / stats['total_day_time']
        stats['idle_pct'] = stats['idle'] / stats['total_day_time']

        result = stats[['date', 'image_device_id', 'total_day_time', 'active', 'idle', 'buffer','active_pct', 'idle_pct', 'buffer_pct']]
        pd.testing.assert_frame_equal(result, expected, check_like=True)

    def test_start_and_end_overflow(self):
        terminplanner_dummy = pd.DataFrame([
            {
                'Terminbuch': 'MR1',
                'Wochentag': 'MO',
                'TERMINRASTER_NAME': 'not_required',
                'gültig von': pd.to_datetime(day(-10), format='%d.%m.%Y'),
                'gültig bis': pd.to_datetime(day(10), format='%d.%m.%Y'),
                'Termin': '14:00',
                'Dauer in Min.': 120
            }
        ])
        dicom_data_dummy = pd.DataFrame([
            {
                'AccessionNumber': 1,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=13) + pd.Timedelta(minutes=50),
                'image_end': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=25)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=55),
                'image_end': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=30)
            },
            {
                'AccessionNumber': 3,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=45),
                'image_end': day(num_days_from_start=6, hour=16) + pd.Timedelta(minutes=15)
            }
        ])

        expected_total_day_time1 = 2.0 + (25/60)
        expected_active_time1 = 35 + 35 + 30
        expected_buffer_time1 = 5 + 10 + 5
        expected_idle_time1 = 20 + 5

        expected = pd.DataFrame([{
            'date': day(num_days_from_start=6, hour=0),
            'image_device_id': 1,
            'total_day_time': expected_total_day_time1,  # hours
            'active': expected_active_time1 / 60,  # fraction of hours
            'idle': expected_idle_time1 / 60,  # fraction of hours
            'buffer': expected_buffer_time1 / 60,  # fraction of hours
            'active_pct': (expected_active_time1 / 60) / expected_total_day_time1,
            'idle_pct': (expected_idle_time1 / 60) / expected_total_day_time1,
            'buffer_pct': (expected_buffer_time1 / 60) / expected_total_day_time1
        }
        ])

        tp_agg = aggregate_terminplanner(terminplanner_dummy)
        idle_df = calc_idle_time_gaps(dicom_data_dummy, tp_agg, time_buffer_mins=5)

        appts_and_gaps = calc_appts_and_gaps(idle_df)

        appts_and_gaps['status_duration'] = (appts_and_gaps['end'] - appts_and_gaps['start']) / pd.to_timedelta(1, unit='H')

        daily_stats = appts_and_gaps.groupby(['date', 'image_device_id', 'status']).agg({
            'status_duration': 'sum'
        }).reset_index()
        one_hour = pd.to_timedelta(1, unit='H')
        total_day_times = appts_and_gaps.groupby(['date', 'image_device_id']).agg({
            'start': 'min',
            'end': 'max'
        }).reset_index()
        total_day_times['total_day_time'] = (total_day_times['end'] - total_day_times['start'])/one_hour
        stats = daily_stats.pivot_table(columns='status', values='status_duration', index=['date', 'image_device_id']).reset_index()
        stats = stats.merge(total_day_times, on=['date', 'image_device_id'])
        stats['active'] = stats['total_day_time'] - stats['buffer'] - stats['idle']  # as there might be overlaps in appts, so we don't want to doublecount
        stats['active_pct'] = stats['active'] / stats['total_day_time']
        stats['buffer_pct'] = stats['buffer'] / stats['total_day_time']
        stats['idle_pct'] = stats['idle'] / stats['total_day_time']

        result = stats[['date', 'image_device_id', 'total_day_time', 'active', 'idle', 'buffer','active_pct', 'idle_pct', 'buffer_pct']]
        pd.testing.assert_frame_equal(result, expected, check_like=True)

    def test_start_and_end_overflow_with_buffer_overlap(self):
        terminplanner_dummy = pd.DataFrame([
            {
                'Terminbuch': 'MR1',
                'Wochentag': 'MO',
                'TERMINRASTER_NAME': 'not_required',
                'gültig von': pd.to_datetime(day(-10), format='%d.%m.%Y'),
                'gültig bis': pd.to_datetime(day(10), format='%d.%m.%Y'),
                'Termin': '14:00',
                'Dauer in Min.': 120
            }
        ])
        dicom_data_dummy = pd.DataFrame([
            {
                'AccessionNumber': 1,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=13) + pd.Timedelta(minutes=50),
                'image_end': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=25)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=30),
                'image_end': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=30)
            },
            {
                'AccessionNumber': 3,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=45),
                'image_end': day(num_days_from_start=6, hour=16) + pd.Timedelta(minutes=15)
            }
        ])

        expected_total_day_time1 = 2.0 + (25/60)
        expected_active_time1 = 35 + 60 + 30
        expected_buffer_time1 = 5 + 10
        expected_idle_time1 =  5

        expected = pd.DataFrame([{
            'date': day(num_days_from_start=6, hour=0),
            'image_device_id': 1,
            'total_day_time': expected_total_day_time1,  # hours
            'active': expected_active_time1 / 60,  # fraction of hours
            'idle': expected_idle_time1 / 60,  # fraction of hours
            'buffer': expected_buffer_time1 / 60,  # fraction of hours
            'active_pct': (expected_active_time1 / 60) / expected_total_day_time1,
            'idle_pct': (expected_idle_time1 / 60) / expected_total_day_time1,
            'buffer_pct': (expected_buffer_time1 / 60) / expected_total_day_time1
        }
        ])

        tp_agg = aggregate_terminplanner(terminplanner_dummy)
        idle_df = calc_idle_time_gaps(dicom_data_dummy, tp_agg, time_buffer_mins=5)

        appts_and_gaps = calc_appts_and_gaps(idle_df)

        appts_and_gaps['status_duration'] = (appts_and_gaps['end'] - appts_and_gaps['start']) / pd.to_timedelta(1, unit='H')

        daily_stats = appts_and_gaps.groupby(['date', 'image_device_id', 'status']).agg({
            'status_duration': 'sum'
        }).reset_index()
        one_hour = pd.to_timedelta(1, unit='H')
        total_day_times = appts_and_gaps.groupby(['date', 'image_device_id']).agg({
            'start': 'min',
            'end': 'max'
        }).reset_index()
        total_day_times['total_day_time'] = (total_day_times['end'] - total_day_times['start'])/one_hour
        stats = daily_stats.pivot_table(columns='status', values='status_duration', index=['date', 'image_device_id']).reset_index()
        stats = stats.merge(total_day_times, on=['date', 'image_device_id'])
        stats['active'] = stats['total_day_time'] - stats['buffer'] - stats['idle']  # as there might be overlaps in appts, so we don't want to doublecount
        stats['active_pct'] = stats['active'] / stats['total_day_time']
        stats['buffer_pct'] = stats['buffer'] / stats['total_day_time']
        stats['idle_pct'] = stats['idle'] / stats['total_day_time']

        result = stats[['date', 'image_device_id', 'total_day_time', 'active', 'idle', 'buffer','active_pct', 'idle_pct', 'buffer_pct']]
        pd.testing.assert_frame_equal(result, expected, check_like=True)

    def test_start_and_end_overflow_with_appt_overlap(self):
        terminplanner_dummy = pd.DataFrame([
            {
                'Terminbuch': 'MR1',
                'Wochentag': 'MO',
                'TERMINRASTER_NAME': 'not_required',
                'gültig von': pd.to_datetime(day(-10), format='%d.%m.%Y'),
                'gültig bis': pd.to_datetime(day(10), format='%d.%m.%Y'),
                'Termin': '14:00',
                'Dauer in Min.': 120
            }
        ])
        dicom_data_dummy = pd.DataFrame([
            {
                'AccessionNumber': 1,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=13) + pd.Timedelta(minutes=50),
                'image_end': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=25)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=20),
                'image_end': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=30)
            },
            {
                'AccessionNumber': 3,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=45),
                'image_end': day(num_days_from_start=6, hour=16) + pd.Timedelta(minutes=15)
            }
        ])

        expected_total_day_time1 = 2.0 + (25/60)
        expected_active_time1 = 35 + 65 + 30
        expected_buffer_time1 = 10
        expected_idle_time1 = 5

        expected = pd.DataFrame([{
            'date': day(num_days_from_start=6, hour=0),
            'image_device_id': 1,
            'total_day_time': expected_total_day_time1,  # hours
            'active': expected_active_time1 / 60,  # fraction of hours
            'idle': expected_idle_time1 / 60,  # fraction of hours
            'buffer': expected_buffer_time1 / 60,  # fraction of hours
            'active_pct': (expected_active_time1 / 60) / expected_total_day_time1,
            'idle_pct': (expected_idle_time1 / 60) / expected_total_day_time1,
            'buffer_pct': (expected_buffer_time1 / 60) / expected_total_day_time1
        }
        ])

        tp_agg = aggregate_terminplanner(terminplanner_dummy)
        idle_df = calc_idle_time_gaps(dicom_data_dummy, tp_agg, time_buffer_mins=5)

        appts_and_gaps = calc_appts_and_gaps(idle_df)

        stats = calc_daily_idle_time_stats(appts_and_gaps)

        result = stats[['date', 'image_device_id', 'total_day_time', 'active', 'idle', 'buffer',
                        'active_pct', 'idle_pct', 'buffer_pct']]
        pd.testing.assert_frame_equal(result, expected, check_like=True)

    def test_appt_outside_day(self):
        terminplanner_dummy = pd.DataFrame([
            {
                'Terminbuch': 'MR1',
                'Wochentag': 'MO',
                'TERMINRASTER_NAME': 'not_required',
                'gültig von': pd.to_datetime(day(-10), format='%d.%m.%Y'),
                'gültig bis': pd.to_datetime(day(10), format='%d.%m.%Y'),
                'Termin': '14:00',
                'Dauer in Min.': 120
            }
        ])
        dicom_data_dummy = pd.DataFrame([
            {
                'AccessionNumber': 1,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=14) + pd.Timedelta(minutes=30),
                'image_end': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=5)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=20),
                'image_end': day(num_days_from_start=6, hour=15) + pd.Timedelta(minutes=50)
            },
            {
                'AccessionNumber': 3,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=16) + pd.Timedelta(minutes=20),
                'image_end': day(num_days_from_start=6, hour=16) + pd.Timedelta(minutes=50)
            }
        ])

        expected_total_day_time = 2.0
        expected_active_time = 35 + 30
        expected_buffer_time = 10 + 10
        expected_idle_time = 35

        expected = pd.DataFrame([{
            'total_day_time': expected_total_day_time,  # hours
            'active': expected_active_time / 60,  # fraction of hours
            'idle': expected_idle_time / 60,  # fraction of hours
            'buffer': expected_buffer_time / 60,  # fraction of hours
            'active_pct': (expected_active_time / 60) / expected_total_day_time,
            'idle_pct': (expected_idle_time / 60) / expected_total_day_time,
            'buffer_pct': (expected_buffer_time / 60) / expected_total_day_time
        }])

        tp_agg = aggregate_terminplanner(terminplanner_dummy)
        idle_df = calc_idle_time_gaps(dicom_data_dummy, tp_agg, time_buffer_mins=5)

        # stats = calc_daily_idle_time_stats(idle_df)
        # result = stats[['total_day_time', 'active_time', 'idle_time', 'buffer_time', 'active_time_pct',
        #                 'idle_time_pct', 'buffer_time_pct']]
        appts_and_gaps = calc_appts_and_gaps(idle_df)

        appts_and_gaps['status_duration'] = (appts_and_gaps['end'] - appts_and_gaps['start']) / pd.to_timedelta(1,
                                                                                                                unit='H')
        stats = appts_and_gaps.groupby('status').agg({'status_duration': 'sum'})
        total_day_time = (stats.loc['active'] + stats.loc['idle'] + stats.loc['buffer'])[0].astype(float)
        result = pd.DataFrame([{
            'total_day_time': total_day_time,  # hours
            'active': stats.loc['active'][0],  # fraction of hours
            'idle': stats.loc['idle'][0],  # fraction of hours
            'buffer': stats.loc['buffer'][0],  # fraction of hours
            'active_pct': (stats.loc['active'][0]) / total_day_time,
            'idle_pct': (stats.loc['idle'][0]) / total_day_time,
            'buffer_pct': (stats.loc['buffer'][0]) / total_day_time
        }])

        pd.testing.assert_frame_equal(result, expected, check_like=True)
