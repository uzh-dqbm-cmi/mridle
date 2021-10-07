"""
Generate figures for the analysis of the appointment reconstruction algorithm ("data validation"),
 which compares reconstructed RIS data to  schedule data collected manually from the scheduling office ("dispo").

Run the following code in a notebook to view figures (Jaccard Scores table, Confusion Matrices, and Schedule Plots):
```
from mridle.figures import data_validation_results as dvr

data = dvr.load_data()
dispo_data, ris_data = data

# Jaccard scores
jaccard_scores = dvs.calc_jaccard_score_table(data)
jaccard_scores.style.format('{:.3f}')

# Confusion Matrices
dvr.calc_exp_confusion_matrix('development', data)
dvr.calc_exp_confusion_matrix('evaluation', data)

# Schedule Plots
display(dvr.plot_validation_week(dispo_data['slot_df']['development'], 'MR1
, title='Development Set - MR1))
display(dvr.plot_validation_week(dispo_data['slot_df']['development'], 'MR2
, title='Development Set - MR2))

display(dvr.plot_validation_week(dispo_data['slot_df']['evaluation'], 'MR1
, title='Development Set - MR1))
display(dvr.plot_validation_week(dispo_data['slot_df']['evaluation'], 'MR2
, title='Development Set - MR2))
```
"""

import altair as alt
import mridle
import pandas as pd
import numpy as np
import datatc as dtc
from typing import Dict, Tuple, Union
import pandas.io.formats.style

ValDataDict = Dict[str, Dict[str, Union[Dict, pd.DataFrame]]]
ValDataTuple = Tuple[ValDataDict, ValDataDict]


experiments = {
        'development': 2,
        'evaluation': 3,
    }


def load_data() -> ValDataTuple:
    """
    Load the data required to analyze the schedule reconstruction data validation results.

    Returns: Data object containing the Dispo and RIS data, snapshotted at multiple stages of transformation,
    and for both the development and evaluation experiments.
    For Dispo data, that includes the list of records, dataframe of records, and slot dataframe.
    For RIS data, that includes the raw dataframe, status dataframe, and slot dataframe.

    Returns: Data object of type ValDataTuple.

    """
    dd = dtc.DataDirectory.load('/data/mridle/data')

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


def plot_validation_week(dispo_df: pd.DataFrame, machine_id: str = 'MR1', title: str = '') -> alt.Chart:
    """
    Plot the schedule of Dispo appointments in either the Development or Evaluation set.

    Args:
        dispo_df: dispo slot dataframe.
        machine_id: Id of the machine to plot. Either 'MR1' or 'MR2'.
        title: Title for the plot.

    Returns: altair Chart.

    """
    df = dispo_df.copy()
    df = df[(~df['slot_outcome'].isnull()) & (~df['NoShow'].isnull())]
    df['EnteringOrganisationDeviceID'] = np.where(df['machine_after'].isnull(), df['machine_before'],
                                                  df['machine_after'])
    valid_machines = df['EnteringOrganisationDeviceID'].unique()

    if machine_id not in valid_machines:
        raise ValueError(f'machine if {machine_id} not found. Specify one of {valid_machines}')
    df = df[df['EnteringOrganisationDeviceID'] == machine_id]

    df['end_time'] = df['start_time'] + pd.Timedelta(minutes=30)
    df['dayofweek'] = df['date'].dt.day_name()

    outcome_color_map = {
        'show': '#1f77b4',
        'rescheduled': '#ff7f0e',
        'canceled': '#d62728',
    }

    color_scale = alt.Scale(domain=list(outcome_color_map.keys()), range=list(outcome_color_map.values()))

    chart = alt.Chart(df).mark_bar().encode(
        alt.Color('slot_outcome:N', scale=color_scale),
        x='NoShow:N',
        y=alt.Y('hoursminutes(start_time):T', title='Time'),
        y2='hoursminutes(end_time):T',
        column=alt.Column('dayofweek', sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], title=''),
    ).properties(
        width=100,
        title=title
    )
    return chart
