import altair as alt
import mridle
import pandas as pd
import numpy as np
import datatc as dtc
from typing import Dict, Tuple


experiments = {
        'development': 2,
        'evaluation': 3,
    }


def color_red(val):
    if val > 0:
        return 'color: red'
    else:
        return 'color: black'


def color_orange(val):
    if val > 0:
        return 'color: orange'
    else:
        return 'color: black'


def color_green(val):
    if val > 0:
        return 'color: green'
    else:
        return 'color: black'


def load_data() -> Tuple[Dict, Dict]:
    dd = dtc.DataDirectory.load('mridle')

    dispo_data = {
        'records': {
            'development': {},
            'evaluation': {},
        },
        'records_df': {
            'development': pd.DataFrame(),
            'evaluation': pd.DataFrame(),
        },
        'slot_df': {
            'development': pd.DataFrame(),
            'evaluation': pd.DataFrame(),
        },
    }

    test_pat_ids = dd['dispo_data']['test_patient_ids.yaml'].load()

    for exp in experiments:
        exp_no = experiments[exp]
        dispo_records = dd['dispo_data'][f'experiment{exp_no}.yaml'].load()
        dispo_records_corrections = dd['dispo_data'][f'experiment{exp_no}_corrections.yaml'].load()
        dispo_records = dispo_records + dispo_records_corrections
        dispo_data['records'][exp] = dispo_records

        records_df = pd.DataFrame(dispo_records)
        records_df['date_recorded'] = pd.to_datetime(records_df['date_recorded'], dayfirst=True)
        dispo_data['records_df'][exp] = records_df

        dispo_data['slot_df'][exp] = mridle.data_management.build_dispo_exp_2_df(dispo_records, test_pat_ids)

    ris_data = {
        'raw_df': {
            'development': pd.DataFrame(),
            'evaluation': pd.DataFrame(),
        },
        'status_df': {
            'development': pd.DataFrame(),
            'evaluation': pd.DataFrame(),
        },
        'slot_df': {
            'development': pd.DataFrame(),
            'evaluation': pd.DataFrame(),
        },
    }

    ris_data['raw_df']['development'] = dd['rdsc_extracts']['2020_10_13_exp_2_week'][
        'RIS_2020_week40_fix_column_headers.csv'].load()
    ris_data['raw_df']['evaluation'] = dd['rdsc_extracts']['2021_05_11_exp_3_week']['RIS_2021_week15.xlsx'].load()

    for exp in experiments:
        ris_data['status_df'][exp] = mridle.data_management.build_status_df(ris_data['raw_df'][exp], test_pat_ids)
        ris_data['slot_df'][exp] = mridle.data_management.build_slot_df(ris_data['status_df'][exp])
        # Restrict to `MR1` and `MR2` because Dispo data for `MRDL` wasn't collected
        ris_data['slot_df'][exp] = ris_data['slot_df'][exp][
            ris_data['slot_df'][exp]['EnteringOrganisationDeviceID'].isin(['MR1', 'MR2'])].copy()

    return dispo_data, ris_data


def calc_jaccard_score_table(data) -> pd.DataFrame:
    dispo_data, ris_data = data
    jaccard_results = {
        'development': {
            'show': 0,
            'canceled': 0,
            'rescheduled': 0,
        },
        'evaluation': {
            'show': 0,
            'canceled': 0,
            'rescheduled': 0,
        }
    }

    for exp in experiments:
        dispo_slot_df = dispo_data['slot_df'][exp]
        ris_slot_df = ris_data['slot_df'][exp]
        for appt_type in ['show', 'canceled', 'rescheduled']:
            jaccard_results[exp][appt_type] = mridle.data_management.jaccard_for_outcome(dispo_slot_df, ris_slot_df,
                                                                                         appt_type)

    jaccard_results_df = pd.DataFrame(jaccard_results)
    return jaccard_results_df


def calc_exp_confusion_matrix(exp, data):
    """Create a styled dataframe of the confusion matrix for either the development or evaluation experiment."""
    dispo_data, ris_data = data
    if exp not in experiments:
        raise ValueError(f'`exp` must be one of {list(experiments.keys())}')

    c = mridle.data_management.validation_exp_confusion_matrix(dispo_data['slot_df'][exp], ris_data['slot_df'][exp],
                                                               ['show', 'rescheduled', 'canceled']).fillna(0).astype(
        int)
    c.loc['not present', 'missing'] = 0
    c = c.style.applymap(color_red, subset=pd.IndexSlice['rescheduled', ['show']])
    c = c.applymap(color_red, subset=pd.IndexSlice['show', ['rescheduled']])
    c = c.applymap(color_red, subset=pd.IndexSlice['show', ['canceled']])
    c = c.applymap(color_orange, subset=pd.IndexSlice['not present', ])
    c = c.applymap(color_orange, subset=pd.IndexSlice[:, ['missing']])
    c = c.applymap(color_green, subset=pd.IndexSlice['show', ['show']])
    c = c.applymap(color_green, subset=pd.IndexSlice['rescheduled', ['rescheduled']])
    c = c.applymap(color_green, subset=pd.IndexSlice['canceled', ['canceled']])
    return c


def plot_validation_week(df, machine_id='MR1', title=''):
    df = df.copy()
    df = df[(~df['slot_outcome'].isnull()) & (~df['NoShow'].isnull())]
    df['EnteringOrganisationDeviceID'] = np.where(df['machine_after'].isnull(), df['machine_before'],
                                                  df['machine_after'])
    valid_machines = df['EnteringOrganisationDeviceID'].unique()

    if machine_id not in valid_machines:
        raise ValueError(f'machine if {machine_id} not found. Specify one of {valid_machines}')
    df = df[df['EnteringOrganisationDeviceID'] == machine_id]

    df['end_time'] = df['start_time'] + pd.Timedelta(minutes=30)
    df['dayofweek'] = df['date'].dt.day_name()

    OUTCOME_COLOR_MAP = {
        'show': '#1f77b4',
        'rescheduled': '#ff7f0e',
        'canceled': '#d62728',
    }

    color_scale = alt.Scale(domain=list(OUTCOME_COLOR_MAP.keys()), range=list(OUTCOME_COLOR_MAP.values()))

    chart = alt.Chart(df).mark_bar().encode(
        alt.Color('slot_outcome:N', scale=color_scale),
        x='NoShow:N',
        y=alt.Y('hoursminutes(start_time):T', title='Time'),
        y2='hoursminutes(end_time):T',
        column=alt.Column('dayofweek', sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], title=''),
        #     tooltip='patient_id'
    ).configure_mark(
        #     opacity=0.5,
    ).properties(
        width=100,
        #   height=recommended_height,
        title=title
    )
    return chart
