import unittest
import pandas as pd
from mridle.data_management import find_no_shows, set_no_show_type

# threshold = 2
# ok_was_status_changes = ['requested']
# ok_now_status_changes = ['requested', 'registered', 'written', 'waiting', 'started', 'examined']
# relevant_columns = ['date', 'was_sched_for_date', 'was_status', 'now_status']
# for col in relevant_columns:
#     if pd.isnull(row[col]):
#         return False
# if row['PatientClass'] == 'ambulant' \
#         and row['was_sched_for_date'] - row['date'] < pd.Timedelta(days=threshold) \
#         and row['now_status'] not in ok_now_status_changes \
#         and row['was_status'] not in ok_was_status_changes:
#     return True


class TestFindNoShowsPositive(unittest.TestCase):

    def test_rescheduled_on_same_day(self):
        example_row = pd.DataFrame({
            'PatientClass': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'scheduled',
            'now_status': 'scheduled',
        }, index=[0])
        self.assertTrue(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_rescheduled_one_day_prior(self):
        example_row = pd.DataFrame({
            'PatientClass': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=2, hour=9),
            'was_status': 'scheduled',
            'now_status': 'scheduled',
        }, index=[0])
        self.assertTrue(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_registered_to_scheduled_on_same_day(self):
        example_row = pd.DataFrame({
            'PatientClass': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'registered',
            'now_status': 'scheduled',
        }, index=[0])
        self.assertTrue(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_registered_to_cancelled_on_same_day(self):
        example_row = pd.DataFrame({
            'PatientClass': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'registered',
            'now_status': 'cancelled',
        }, index=[0])
        self.assertTrue(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_scheduled_to_cancelled_one_day_prior(self):
        example_row = pd.DataFrame({
            'PatientClass': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=2, hour=9),
            'was_status': 'scheduled',
            'now_status': 'cancelled',
        }, index=[0])
        self.assertTrue(example_row.apply(find_no_shows, axis=1).iloc[0])


class TestFindNoShowsNegative(unittest.TestCase):

    def test_rescheduled_three_days_in_advance(self):
        example_row = pd.DataFrame({
            'PatientClass': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=4, hour=9),
            'was_status': 'scheduled',
            'now_status': 'scheduled',
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_rescheduled_registered_to_scheduled_three_days_in_advance(self):
        example_row = pd.DataFrame({
            'PatientClass': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=4, hour=9),
            'was_status': 'registered',
            'now_status': 'scheduled',
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_stationar_rescheduled_same_day(self):
        example_row = pd.DataFrame({
            'PatientClass': 'stationär',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'scheduled',
            'now_status': 'scheduled',
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_teilstationar_rescheduled_same_day(self):
        example_row = pd.DataFrame({
            'PatientClass': 'teilstationär',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'scheduled',
            'now_status': 'scheduled',
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_changed_from_requested_to_scheduled_same_day(self):
        example_row = pd.DataFrame({
            'PatientClass': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'requested',
            'now_status': 'scheduled',
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_changed_from_requested_to_requested_same_day(self):
        example_row = pd.DataFrame({
            'PatientClass': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'requested',
            'now_status': 'requested',
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_changed_from_scheduled_to_registered_same_day(self):
        example_row = pd.DataFrame({
            'PatientClass': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'scheduled',
            'now_status': 'registered',
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_changed_from_registered_to_waiting_same_day(self):
        example_row = pd.DataFrame({
            'PatientClass': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'registered',
            'now_status': 'waiting',
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_changed_from_waiting_to_started_same_day(self):
        example_row = pd.DataFrame({
            'PatientClass': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'waiting',
            'now_status': 'started',
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_changed_from_started_to_examined_same_day(self):
        example_row = pd.DataFrame({
            'PatientClass': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'started',
            'now_status': 'examined',
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_changed_from_scheduled_to_written_same_day(self):
        example_row = pd.DataFrame({
            'PatientClass': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': 'scheduled',
            'now_status': 'written',
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_no_was_status(self):
        example_row = pd.DataFrame({
            'PatientClass': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_status': None,
            'now_status': 'scheduled',
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_no_was_sched_for_date(self):
        example_row = pd.DataFrame({
            'PatientClass': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1),
            'was_sched_for_date': None,
            'was_status': 'scheduled',
            'now_status': 'scheduled',
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])

    def test_was_sched_for_date_is_midnight(self):
        example_row = pd.DataFrame({
            'PatientClass': 'ambulant',
            'date': pd.Timestamp(year=2019, month=1, day=1, hour=9),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=2, hour=0),
            'was_status': 'scheduled',
            'now_status': 'scheduled',
        }, index=[0])
        self.assertFalse(example_row.apply(find_no_shows, axis=1).iloc[0])


class TestSetNoShowType(unittest.TestCase):

    def test_not_no_show_blank(self):
        example_row = pd.DataFrame({
            'NoShow': False,
        }, index=[0])
        expected_result = None
        self.assertEquals(example_row.apply(set_no_show_type, axis=1).iloc[0], expected_result)

    def test_rescheduled_two_days_in_advance_soft(self):
        example_row = pd.DataFrame({
            'date': pd.Timestamp(year=2019, month=1, day=2),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=4, hour=9),
            'NoShow': True,
        }, index=[0])
        expected_result = 'soft'
        self.assertEquals(example_row.apply(set_no_show_type, axis=1).iloc[0], expected_result)

    def test_rescheduled_one_day_in_advance_soft(self):
        example_row = pd.DataFrame({
            'date': pd.Timestamp(year=2019, month=1, day=3),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=4, hour=9),
            'NoShow': True,
        }, index=[0])
        expected_result = 'soft'
        self.assertEquals(example_row.apply(set_no_show_type, axis=1).iloc[0], expected_result)

    def test_rescheduled_one_hour_later_hard(self):
        example_row = pd.DataFrame({
            'date': pd.Timestamp(year=2019, month=1, day=4, hour=10),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=4, hour=9),
            'NoShow': True,
        }, index=[0])
        expected_result = 'hard'
        self.assertEquals(example_row.apply(set_no_show_type, axis=1).iloc[0], expected_result)

    def test_rescheduled_one_day_later_hard(self):
        example_row = pd.DataFrame({
            'date': pd.Timestamp(year=2019, month=1, day=5, hour=9),
            'was_sched_for_date': pd.Timestamp(year=2019, month=1, day=4, hour=9),
            'NoShow': True,
        }, index=[0])
        expected_result = 'hard'
        self.assertEquals(example_row.apply(set_no_show_type, axis=1).iloc[0], expected_result)
