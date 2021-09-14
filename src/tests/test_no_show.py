import unittest
import pandas as pd
from mridle.data_management import find_no_shows, set_no_show_severity


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
