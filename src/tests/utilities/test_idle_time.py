import unittest
import pandas as pd
from mridle.utilities.data_management import aggregate_terminplanner
from mridle.utilities.idle_time import calc_idle_time_gaps, calc_daily_idle_time_stats, calc_appts_and_gaps


def day(num_days_from_start, hour=9, minute=0):
    return pd.Timestamp(year=2019, month=1, day=1, hour=hour, minute=minute) + pd.Timedelta(days=num_days_from_start)


one_hour = pd.to_timedelta(1, unit='H')


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
                'image_start': day(num_days_from_start=6, hour=14, minute=30),
                'image_end': day(num_days_from_start=6, hour=15, minute=5)
            }
        ])

        expected_total_day_time = 2.0
        expected_active_time = 35
        expected_buffer_time = 10
        expected_idle_time = 75

        expected = pd.DataFrame([{
            'date': day(num_days_from_start=6, hour=0),
            'image_device_id': 1,
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

        appts_and_gaps = calc_appts_and_gaps(idle_df)

        stats = calc_daily_idle_time_stats(appts_and_gaps)
        result = stats[['date', 'image_device_id', 'total_day_time', 'active', 'idle', 'buffer',
                        'active_pct', 'idle_pct', 'buffer_pct']]
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
                'image_start': day(num_days_from_start=6, hour=14, minute=30),
                'image_end': day(num_days_from_start=6, hour=15, minute=5)
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
            'date': day(num_days_from_start=6, hour=0),
            'image_device_id': 1,
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

        appts_and_gaps = calc_appts_and_gaps(idle_df)

        stats = calc_daily_idle_time_stats(appts_and_gaps)
        result = stats[['date', 'image_device_id', 'total_day_time', 'active', 'idle', 'buffer',
                        'active_pct', 'idle_pct', 'buffer_pct']]

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
                'image_start': day(num_days_from_start=6, hour=14, minute=30),
                'image_end': day(num_days_from_start=6, hour=15, minute=5)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=15, minute=20),
                'image_end': day(num_days_from_start=6, hour=15, minute=50)
            },
            {
                'AccessionNumber': 3,
                'StudyDescription': '-',
                'image_device_id': 2,
                'image_start': day(num_days_from_start=6, hour=14, minute=20),
                'image_end': day(num_days_from_start=6, hour=14, minute=45)
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
            'date': day(num_days_from_start=6, hour=0),
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

        appts_and_gaps = calc_appts_and_gaps(idle_df)

        stats = calc_daily_idle_time_stats(appts_and_gaps)
        result = stats[['date', 'image_device_id', 'total_day_time', 'active', 'idle', 'buffer',
                        'active_pct', 'idle_pct', 'buffer_pct']]
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
                'image_start': day(num_days_from_start=6, hour=14, minute=20),
                'image_end': day(num_days_from_start=6, hour=14, minute=50)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=14, minute=55),
                'image_end': day(num_days_from_start=6, hour=15, minute=15)
            },
            {
                'AccessionNumber': 3,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=15, minute=20),
                'image_end': day(num_days_from_start=6, hour=15, minute=50)
            }
        ])

        expected_total_day_time1 = 2.0
        expected_active_time1 = 80
        expected_buffer_time1 = 20
        expected_idle_time1 = 20

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
                'image_start': day(num_days_from_start=6, hour=14, minute=20),
                'image_end': day(num_days_from_start=6, hour=14, minute=50)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=14, minute=45),
                'image_end': day(num_days_from_start=6, hour=15, minute=15)
            }
        ])

        expected_total_day_time1 = 2.0
        expected_active_time1 = 55
        expected_buffer_time1 = 10
        expected_idle_time1 = 55

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
                'image_start': day(num_days_from_start=6, hour=13, minute=55),
                'image_end': day(num_days_from_start=6, hour=14, minute=20)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=14, minute=45),
                'image_end': day(num_days_from_start=6, hour=15, minute=15)
            }
        ])

        expected_total_day_time1 = 2.0 + 5 / 60
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

        appts_and_gaps = calc_appts_and_gaps(idle_df)

        stats = calc_daily_idle_time_stats(appts_and_gaps)

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
                'image_start': day(num_days_from_start=6, hour=14, minute=20),
                'image_end': day(num_days_from_start=6, hour=14, minute=55)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=15, minute=35),
                'image_end': day(num_days_from_start=6, hour=16, minute=10)
            }
        ])

        expected_total_day_time1 = 2.0 + 10 / 60
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

        appts_and_gaps = calc_appts_and_gaps(idle_df)

        stats = calc_daily_idle_time_stats(appts_and_gaps)

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
                'image_start': day(num_days_from_start=6, hour=14, minute=30),
                'image_end': day(num_days_from_start=6, hour=15, minute=5)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=15, minute=20),
                'image_end': day(num_days_from_start=6, hour=15, minute=50)
            },
            {
                'AccessionNumber': 3,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=7, hour=14, minute=20),
                'image_end': day(num_days_from_start=7, hour=14, minute=45)
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

        appts_and_gaps = calc_appts_and_gaps(idle_df)

        stats = calc_daily_idle_time_stats(appts_and_gaps)

        result = stats[
            ['date', 'image_device_id', 'total_day_time', 'active', 'idle', 'buffer', 'active_pct', 'idle_pct',
             'buffer_pct']]
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
                'image_start': day(num_days_from_start=6, hour=13, minute=50),
                'image_end': day(num_days_from_start=6, hour=14, minute=25)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=14, minute=55),
                'image_end': day(num_days_from_start=6, hour=15, minute=30)
            },
            {
                'AccessionNumber': 3,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=15, minute=45),
                'image_end': day(num_days_from_start=6, hour=16, minute=15)
            }
        ])

        expected_total_day_time1 = 2.0 + (25 / 60)
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

        stats = calc_daily_idle_time_stats(appts_and_gaps)

        result = stats[
            ['date', 'image_device_id', 'total_day_time', 'active', 'idle', 'buffer', 'active_pct', 'idle_pct',
             'buffer_pct']]
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
                'image_start': day(num_days_from_start=6, hour=13, minute=50),
                'image_end': day(num_days_from_start=6, hour=14, minute=25)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=14, minute=30),
                'image_end': day(num_days_from_start=6, hour=15, minute=30)
            },
            {
                'AccessionNumber': 3,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=15, minute=45),
                'image_end': day(num_days_from_start=6, hour=16, minute=15)
            }
        ])

        expected_total_day_time1 = 2.0 + (25 / 60)
        expected_active_time1 = 35 + 60 + 30
        expected_buffer_time1 = 5 + 10
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

        result = stats[
            ['date', 'image_device_id', 'total_day_time', 'active', 'idle', 'buffer', 'active_pct', 'idle_pct',
             'buffer_pct']]
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
                'image_start': day(num_days_from_start=6, hour=13, minute=50),
                'image_end': day(num_days_from_start=6, hour=14, minute=25)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=14, minute=20),
                'image_end': day(num_days_from_start=6, hour=15, minute=30)
            },
            {
                'AccessionNumber': 3,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=15, minute=45),
                'image_end': day(num_days_from_start=6, hour=16, minute=15)
            }
        ])

        expected_total_day_time1 = 2.0 + (25 / 60)
        expected_active_time1 = 35 + 70 - 5 + 30
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
                'image_start': day(num_days_from_start=6, hour=14, minute=30),
                'image_end': day(num_days_from_start=6, hour=15, minute=5)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=15, minute=20),
                'image_end': day(num_days_from_start=6, hour=15, minute=50)
            },
            {
                'AccessionNumber': 3,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=16, minute=20),
                'image_end': day(num_days_from_start=6, hour=16, minute=50)
            }
        ])

        expected_total_day_time = 2.0
        expected_active_time = 35 + 30
        expected_buffer_time = 10 + 10
        expected_idle_time = 35

        expected = pd.DataFrame([{
            'date': day(num_days_from_start=6, hour=0),
            'image_device_id': 1,
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

        appts_and_gaps = calc_appts_and_gaps(idle_df)

        stats = calc_daily_idle_time_stats(appts_and_gaps)
        result = stats[['date', 'image_device_id', 'total_day_time', 'active', 'idle', 'buffer',
                        'active_pct', 'idle_pct', 'buffer_pct']]

        pd.testing.assert_frame_equal(result, expected, check_like=True)

    def test_no_buffer(self):
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
                'image_start': day(num_days_from_start=6, hour=14, minute=30),
                'image_end': day(num_days_from_start=6, hour=15, minute=5)
            },
            {
                'AccessionNumber': 2,
                'StudyDescription': '-',
                'image_device_id': 1,
                'image_start': day(num_days_from_start=6, hour=15, minute=20),
                'image_end': day(num_days_from_start=6, hour=15, minute=50)
            }
        ])

        expected_total_day_time = 2.0
        expected_active_time = 35 + 30
        expected_buffer_time = 0
        expected_idle_time = 55

        expected = pd.DataFrame([{
            'date': day(num_days_from_start=6, hour=0),
            'image_device_id': 1,
            'total_day_time': expected_total_day_time,  # hours
            'active': expected_active_time / 60,  # fraction of hours
            'idle': expected_idle_time / 60,  # fraction of hours
            'buffer': expected_buffer_time / 60,  # fraction of hours
            'active_pct': (expected_active_time / 60) / expected_total_day_time,
            'idle_pct': (expected_idle_time / 60) / expected_total_day_time,
            'buffer_pct': (expected_buffer_time / 60) / expected_total_day_time
        }])

        tp_agg = aggregate_terminplanner(terminplanner_dummy)
        idle_df = calc_idle_time_gaps(dicom_data_dummy, tp_agg, time_buffer_mins=0)

        appts_and_gaps = calc_appts_and_gaps(idle_df)

        stats = calc_daily_idle_time_stats(appts_and_gaps)
        result = stats[['date', 'image_device_id', 'total_day_time', 'active', 'idle', 'buffer',
                        'active_pct', 'idle_pct', 'buffer_pct']]

        pd.testing.assert_frame_equal(result, expected, check_like=True)
