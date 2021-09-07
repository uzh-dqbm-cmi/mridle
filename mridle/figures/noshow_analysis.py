import pandas as pd
import altair as alt
import plotly.express as px
import numpy as np

alt.data_transformers.disable_max_rows()


def plot_day_no_show(df):
    """

    Args:
        df:

    Returns:

    """
    df_day_agg = df.copy()
    df_day_agg = df_day_agg[df_day_agg['day_of_week_str'].isin(['Monday', 'Tuesday', 'Wednesday',
                                                                'Thursday', 'Friday'])]
    df_day_agg = df_day_agg[['day_of_week_str', 'NoShow']].groupby(['day_of_week_str']).apply(
        lambda x: np.sum(x) / len(x)).reset_index()

    return alt.Chart(df_day_agg).mark_bar().encode(
        alt.X('day_of_week_str', sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']),
        alt.Y('NoShow', axis=alt.Axis(format='%'))
    ).properties(width=250, title='Day of Week')


def plot_month_no_show(df):
    """

    Args:
        df:

    Returns:

    """
    df_month_agg = df.copy()
    df_month_agg = df_month_agg[['month', 'NoShow']].groupby('month').apply(lambda x: np.sum(x) / len(x))
    return alt.Chart(df_month_agg).mark_bar(color='#409caf').encode(
        alt.X('month:O'),
        alt.Y('NoShow', axis=alt.Axis(format='%'))
    ).properties(width=250, title='Month')


def plot_hour_no_show(df):
    """

    Args:
        df:

    Returns:

    """
    df_hour_agg = df.copy()
    df_hour_agg = df_hour_agg[(df_hour_agg['hour_sched'] > 6) & (df_hour_agg['hour_sched'] < 18)]
    df_hour_agg = df_hour_agg[['hour_sched', 'NoShow']].groupby('hour_sched').apply(lambda x: np.sum(x) / len(x))

    return alt.Chart(df_hour_agg).mark_bar(color='#D35400').encode(
        alt.X('hour_sched:O'),
        alt.Y('NoShow', axis=alt.Axis(format='%'))
    ).properties(width=400, title='Hour of Day')


def plot_age_no_show(df):
    """

    Args:
        df:

    Returns:

    """
    df_age_agg = df.copy()
    df_age_agg = df_age_agg[['age', 'NoShow']].groupby('age').agg({'NoShow': ['sum', 'count']}).reset_index()
    df_age_agg.columns = list(map(''.join, df_age_agg.columns.values))
    df_age_agg.columns = ['age', 'NoShowSum', 'Number of patients']

    df_age_agg['NoShow'] = df_age_agg['NoShowSum'] / df_age_agg['Number of patients']
    return alt.Chart(df_age_agg).mark_circle().encode(
        alt.X('age'),
        alt.Y('NoShow', axis=alt.Axis(format='%')),
        size=alt.Size('Number of patients')
    ).properties(width=400)


def plot_appts_per_patient(df, log_scale=False):
    """

    Args:
        df:
        log_scale:

    Returns:

    """

    df_copy = df.copy()
    appts_per_patient = df_copy.groupby('MRNCmpdId').agg({'NoShow': ['count', 'sum']}).reset_index()
    appts_per_patient.columns = ['MRNCmpdId', 'num_appts', 'num_noshow']

    appts_per_patient_agg = appts_per_patient['num_appts'].value_counts().reset_index()
    appts_per_patient_agg.columns = ['num_appts', 'num_patients']

    if log_scale:
        chart = alt.Chart(appts_per_patient_agg).mark_bar(size=15).encode(
            x=alt.X('num_appts', axis=alt.Axis(tickCount=appts_per_patient_agg.shape[0], grid=False)),
            y=alt.Y('num_patients', scale=alt.Scale(type='log'))
        ).properties(title='Number of appointments per patient (log scale)')
    else:
        chart = alt.Chart(appts_per_patient_agg).mark_bar(size=15).encode(
            x=alt.X('num_appts', axis=alt.Axis(tickCount=appts_per_patient_agg.shape[0], grid=False)),
            y='num_patients'
        ).properties(title='Number of appointments per patient')

    return chart


def plot_no_show_scatter(df, log_scale=False):
    """

    Args:
        df:
        log_scale:

    Returns:

    """
    df_copy = df.copy()
    pat_appts = df_copy.groupby('MRNCmpdId').agg({'NoShow': ['count', 'sum']}).reset_index()
    pat_appts.columns = ['MRNCmpdId', 'num_appts', 'num_noshow']

    pat_appts_counts = pat_appts.groupby(['num_appts', 'num_noshow']).count().reset_index()
    pat_appts_counts.columns = ['num_appts', 'num_noshow', 'num_patients']
    pat_appts_counts['percent_missed'] = (pat_appts_counts['num_noshow'] / pat_appts_counts['num_appts'])
    pat_appts_counts.sort_values('num_patients', ascending=False)

    if log_scale:
        chart = alt.Chart(pat_appts_counts).mark_rect().encode(
            x='num_appts:O',
            y=alt.Y('num_noshow:O', sort='descending'),
            color=alt.Color('num_patients:Q', scale=alt.Scale(type='log'))
        ).properties(title='Heat map of appointments and no-show distribution (log scale)')
    else:
        chart = alt.Chart(pat_appts_counts).mark_rect().encode(
            x='num_appts:O',
            y=alt.Y('num_noshow:O', sort='descending'),
            color=alt.Color('num_patients:Q')
        ).properties(title='Heat map of appointments and no-show distribution')

    return chart


def plot_appt_noshow_tree_map(df):
    """

    Args:
        df:

    Returns:

    """
    df_copy = df.copy()
    appts_per_patient = df_copy.groupby('MRNCmpdId').agg({'NoShow': ['count', 'sum']}).reset_index()
    appts_per_patient.columns = ['MRNCmpdId', 'num_appts', 'num_noshow']

    pat_appts_counts = appts_per_patient.groupby(['num_appts', 'num_noshow']).count().reset_index()
    pat_appts_counts.columns = ['num_appts', 'num_noshow', 'num_patients']

    pat_appts_counts['num_appts'] = pd.cut(pat_appts_counts['num_appts'], [0, 1.1, 2.1, 1000],
                                           labels=['Number of appointments: 1', '2', '>2'])
    pat_appts_counts['num_noshow'] = pd.cut(pat_appts_counts['num_noshow'], [-0.1, .1, 1.1, 2.1, 1000],
                                            labels=['Number of no-shows: 0', '1', '2', '>2'])
    appts_per_patient_counts = pat_appts_counts[['num_appts', 'num_noshow', 'num_patients']].groupby(
        ['num_appts', 'num_noshow']).sum().reset_index()

    fig = px.treemap(appts_per_patient_counts, path=['num_appts', 'num_noshow'], values='num_patients',
                     color='num_noshow', color_discrete_map={'(?)': 'burlywood', 'Number of no-shows: 0': 'cadetblue',
                                                             '1': 'coral', '2': 'lightcoral', '>2': 'darkorange'})
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

    return fig
