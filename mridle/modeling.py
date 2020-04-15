def calc_scheduled_in_advance(row):
    if row['was_sched_for'] != row['now_sched_for']:
        return row['now_sched_for']
    else:
        return None
