import altair as alt
from copy import deepcopy
import pandas as pd
import numpy as np
import plotly.express as px
import random
from mridle.utilities.plotting_utilities import DEFAULT_COLOR_MAP, DETAILED_COLOR_MAP, OUTCOME_STROKE_MAP
from typing import List

alt.data_transformers.disable_max_rows()


def plot_dave_b(slot_df: pd.DataFrame, slot_w_dicom_df: pd.DataFrame, valid_date_range: List[str],
                example_date: str = None, anonymize: bool = True) -> alt.Chart:
    """
    Create the Dave B figure, which consists of three subplots:
    - an example day (uses slot_w_dicom_df to show actual times, rather than scheduled)
    - aggregate counts of appointments over time
    - patterns in slot frequencies by day of week

    Args:
        slot_df: slot_df
        slot_w_dicom_df: slot_df with start and end times of appointment slots as determined by the dicom images rather
         than RIS.
        example_date: date to represent in the example day subplot. If none, a random date between start_date and
         end_date is chosen.
        valid_date_range: date range for the 2 aggregate plots
        anonymize: Whether to anonymize the example_day subplot

    Returns: altair plot
    """

    start_date, end_date = valid_date_range
    if example_date is None:
        # choose a random date
        random_row = slot_w_dicom_df[(~slot_w_dicom_df['start_time'].isna())
                                     & (slot_w_dicom_df['start_time'] > start_date)
                                     & (slot_w_dicom_df['end_time'] < end_date)
                                     ].sample(1)
        example_date = random_row['start_time'].dt.floor('d').iloc[0]

    example_day = plot_example_day(slot_w_dicom_df, example_date, anonymize=anonymize)
    daily_over_time = plot_appt_types_over_time(slot_df, start_date, end_date)
    day_of_week = plot_appt_types_by_day_of_week(slot_df, start_date, end_date)
    return (example_day & (daily_over_time | day_of_week)).configure_mark(opacity=0.75)


def plot_example_day(df: pd.DataFrame, date, device='MR1', color_map=DETAILED_COLOR_MAP, stroke_map=OUTCOME_STROKE_MAP,
                     anonymize=True, height_factor=20):
    """
    Plot completed, inpatient, and no-show appointments for one device for a day.

    Args:
        df: a one-row-per-slot dataframe.
        device: the device to plot
        date: date to plot
        color_map: the colors to use for each appointment type.
        stroke_map: the colors to use for each slot_outcome type.
        anonymize: whether to anonymize the data by shifting it by a random amount.
        height_factor: multiplier for how tall to make the plot based on the number of days plotted

    Returns: alt.Chart

    """
    df_filtered = df.copy()
    start_date = pd.to_datetime(date)
    end_date = start_date + pd.Timedelta(days=1)
    df_filtered = df_filtered[df_filtered['start_time'] >= start_date]
    df_filtered = df_filtered[df_filtered['start_time'] < end_date]
    if df_filtered.shape[0] == 0:
        raise ValueError('No data found in that date range')

    if device not in df['EnteringOrganisationDeviceID'].unique():
        raise ValueError('Device {} not found in data set'.format(device))
    df_filtered = df_filtered[df_filtered['EnteringOrganisationDeviceID'] == device].copy()

    title = device

    if anonymize:
        randomizer = pd.Timedelta(minutes=random.randint(-15, 15))
        df_filtered['start_time'] = df_filtered['start_time'] + randomizer
        df_filtered['end_time'] = df_filtered['end_time'] + randomizer
        title = 'Anonymized Appointment Data'

    # create color scale from color_map, modifying it based on highlight if appropriate
    plot_color_map = deepcopy(color_map)
    color_scale = alt.Scale(domain=list(plot_color_map.keys()), range=list(plot_color_map.values()))
    stroke_scale = alt.Scale(domain=list(stroke_map.keys()), range=list(stroke_map.values()))

    recommended_height = len(DEFAULT_COLOR_MAP) * height_factor

    return alt.Chart(df_filtered).mark_bar(strokeWidth=3).encode(
        y=alt.Y('slot_type:N', title='Slot Type'),
        x=alt.X('hoursminutes(start_time):T', title='Time'),
        x2=alt.X2('hoursminutes(end_time):T'),
        color=alt.Color('slot_type_detailed:N', scale=color_scale, legend=alt.Legend(title='Slot Type (detailed)')),
        stroke=alt.Stroke('slot_outcome', scale=stroke_scale, legend=alt.Legend(title='Canceled Appts')),
        tooltip='FillerOrderNo',
    ).properties(
        width=800,
        height=recommended_height,
        title=title
    )


def plot_appt_types_over_time(df: pd.DataFrame, start_date, end_date, color_map=DETAILED_COLOR_MAP):
    df_filtered = df.copy()

    # create week column for plotting
    df_filtered['week'] = df_filtered['start_time'] - pd.to_timedelta(df_filtered['start_time'].dt.dayofweek, unit='d')
    df_filtered['week'] = pd.to_datetime(df_filtered['week'].dt.date)

    df_filtered['year_month'] = df_filtered['start_time'].dt.to_period('M').dt.to_timestamp()

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # make sure the end date is at a week boundary
    end_date = end_date - pd.to_timedelta(end_date.dayofweek, unit='d')
    df_filtered = df_filtered[df_filtered['start_time'] >= start_date]
    df_filtered = df_filtered[df_filtered['start_time'] < end_date]
    if df_filtered.shape[0] == 0:
        raise ValueError('No data found in that date range')

    # create color scale from color_map, modifying it based on highlight if appropriate
    plot_color_map = deepcopy(color_map)
    color_scale = alt.Scale(domain=list(plot_color_map.keys()), range=list(plot_color_map.values()))

    return alt.Chart(df_filtered).mark_line(strokeWidth=3).encode(
        y=alt.Y('count(FillerOrderNo):N', title='Appt Count'),
        x=alt.X('yearmonth(year_month):T', title='Week', axis=alt.Axis(format="%b-%Y")),
        order=alt.Order("yearmonth(year_month)"),
        color=alt.Color('slot_type_detailed:N', scale=color_scale, legend=alt.Legend(title='Slot Type (detailed)')),
        tooltip='yearmonth(year_month)',
    ).properties(
        width=475,
        height=250,
        # title=title
    )


def dayofweek_name(x):
    days_of_week = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }
    return days_of_week[x]


def plot_appt_types_by_day_of_week(df: pd.DataFrame, start_date, end_date, color_map=DETAILED_COLOR_MAP):
    df_filtered = df.copy()
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df_filtered = df_filtered[df_filtered['start_time'] >= start_date]
    df_filtered = df_filtered[df_filtered['start_time'] < end_date]
    if df_filtered.shape[0] == 0:
        raise ValueError('No data found in that date range')

    df_filtered['weekday'] = df_filtered['start_time'].dt.dayofweek
    day_of_week_pivot = pd.pivot_table(df_filtered, index='slot_type_detailed', columns='weekday',
                                       values='FillerOrderNo', aggfunc='count')
    day_of_week_pivot_percent = day_of_week_pivot / day_of_week_pivot.sum()
    melted = pd.melt(day_of_week_pivot_percent.reset_index(), id_vars='slot_type_detailed', value_name='percent')
    melted_filtered = melted[melted['weekday'] < 5].copy()
    melted_filtered['weekday'] = melted_filtered['weekday'].apply(dayofweek_name)

    plot_color_map = deepcopy(color_map)
    color_scale = alt.Scale(domain=list(plot_color_map.keys()), range=list(plot_color_map.values()))

    return alt.Chart(melted_filtered).mark_line(strokeWidth=3).encode(
        x=alt.X('weekday', title='Day of the Week', sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']),
        y=alt.Y('percent', title='Percent of Total Appointments', axis=alt.Axis(format='%')),
        color=alt.Color('slot_type_detailed:N', scale=color_scale, legend=alt.Legend(title='Slot Type (detailed)')),
        tooltip='percent',
    ).properties(
        width=250,
        height=250,
        # title=title
    )


def plot_no_show_by_day_of_week(df):
    """
    Plot no show rate aggregated by the day of the week

    Args:
        df: dataframe of appointments, with a day_of_week_str column and an indicator of whether the patient showed up
            for the appointment

    Returns:
        altair bar chart with one bar per day of the week showing the aggregated no-show rate
    """
    df_day_agg = df.copy()
    df_day_agg = df_day_agg[df_day_agg['day_of_week_str'].isin(['Monday', 'Tuesday', 'Wednesday',
                                                                'Thursday', 'Friday'])]
    df_day_agg = df_day_agg[['day_of_week_str', 'NoShow']].groupby(['day_of_week_str']).apply(
        lambda x: np.sum(x) / len(x)).reset_index()

    return alt.Chart(df_day_agg).mark_bar().encode(
        alt.X('day_of_week_str', sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
              title='Day of the Week'),
        alt.Y('NoShow', axis=alt.Axis(format='%'))
    ).properties(width=250)


def plot_no_show_by_month(df):
    """
    Plot no show rate aggregated by the month

    Args:
        df: dataframe of appointments, with a month column and an indicator of whether the patient showed up for the
            appointment

    Returns:
        altair bar chart with one bar per month of the year showing the aggregated no-show rate
    """
    df_month_agg = df.copy()
    df_month_agg = df_month_agg[['month', 'NoShow']].groupby('month').apply(lambda x: np.sum(x) / len(x))
    return alt.Chart(df_month_agg).mark_bar(color='#409caf').encode(
        alt.X('month:O', title='Month'),
        alt.Y('NoShow', axis=alt.Axis(format='%'))
    ).properties(width=250)


def plot_no_show_by_hour_of_day(df):
    """
    Plot no show rate aggregated by the hour of the day

    Args:
        df: dataframe of appointments, with a hour column and an indicator of whether the patient showed up for the
            appointment

    Returns:
        altair bar chart with one bar per hour of the day showing the aggregated no-show rate

    """
    df_hour_agg = df.copy()
    df_hour_agg = df_hour_agg[(df_hour_agg['hour_sched'] > 6) & (df_hour_agg['hour_sched'] < 18)]
    df_hour_agg = df_hour_agg[['hour_sched', 'NoShow']].groupby('hour_sched').apply(lambda x: np.sum(x) / len(x))

    return alt.Chart(df_hour_agg).mark_bar(color='#D35400').encode(
        alt.X('hour_sched:O', title='Hour Scheduled'),
        alt.Y('NoShow', axis=alt.Axis(format='%'))
    ).properties(width=400, title='')


def plot_no_show_by_age(df):
    """
    Plot no show rate aggregated by the age of the patient

    Args:
        df: dataframe of appointments, with an age column and an indicator of whether the patient showed up for the
            appointment

    Returns:
        altair scatter plot with one mark/point per age on the x-axis, and the no-show rate on the y axis. Each point
        is sized relative to the number of patients in that age group

    """
    df_age_agg = df.copy()
    df_age_agg = df_age_agg[['age', 'NoShow']].groupby('age').agg({'NoShow': ['sum', 'count']}).reset_index()
    df_age_agg.columns = list(map(''.join, df_age_agg.columns.values))
    df_age_agg.columns = ['age', 'NoShowSum', 'Number of patients']

    df_age_agg['NoShow'] = df_age_agg['NoShowSum'] / df_age_agg['Number of patients']
    return alt.Chart(df_age_agg).mark_circle().encode(
        alt.X('age', title='Age'),
        alt.Y('NoShow', axis=alt.Axis(format='%')),
        size=alt.Size('Number of patients')
    ).properties(width=400)


def plot_appts_per_patient(df, log_scale=False):
    """
    Plot a histogram showing the number of patients which had 1 appointment, 2 appointments, 3, 4, ... and so on

    Args:
        df: dataframe containing appointments with patient IDs
        log_scale: a boolean of whether to plot the y-axis in the log-scale or not

    Returns:
        a histogram showing the number of patients which had 1 appointment, 2 appointments, 3, 4, ... and so on
    """

    df_copy = df.copy()
    appts_per_patient = df_copy.groupby('MRNCmpdId').agg({'NoShow': ['count', 'sum']}).reset_index()
    appts_per_patient.columns = ['MRNCmpdId', 'num_appts', 'num_noshow']

    appts_per_patient_agg = appts_per_patient['num_appts'].value_counts().reset_index()
    appts_per_patient_agg.columns = ['num_appts', 'num_patients']

    if log_scale:
        chart = alt.Chart(appts_per_patient_agg).mark_bar(size=15).encode(
            x=alt.X('num_appts', title='Number of Appointments',
                    axis=alt.Axis(tickCount=appts_per_patient_agg.shape[0], grid=False)),
            y=alt.Y('num_patients', title='Number of Patients', scale=alt.Scale(type='log'))
        ).properties(title='Number of appointments per patient (log scale)')
    else:
        chart = alt.Chart(appts_per_patient_agg).mark_bar(size=15).encode(
            x=alt.X('num_appts', title='Number of Appointments',
                    axis=alt.Axis(tickCount=appts_per_patient_agg.shape[0], grid=False)),
            y=alt.Y('num_patients', title='Number of Patients',)
        ).properties(title='Number of appointments per patient')

    return chart


def plot_no_show_heat_map(df, log_scale=False):
    """
    Plot a heatmap with number of appointments on the x-axis and number of no-shows on the y-axis, with each section
    coloured by the number of patients in each.

    Args:
        df: dataframe containing appointments with patient IDs and no-show information
        log_scale: a boolean of whether to plot the y-axis in the log-scale or not

    Returns:
        a heatmap with number of appointments on the x-axis and number of no-shows on the y-axis, with each section
        coloured by the number of patients in each.
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
            x=alt.X('num_appts:O', title='Number of Appointments'),
            y=alt.Y('num_noshow:O', title='Number of No-shows', sort='descending'),
            color=alt.Color('num_patients:Q', scale=alt.Scale(type='log'))
        ).properties(title='Heat map of appointments and no-show distribution (log scale)')
    else:
        chart = alt.Chart(pat_appts_counts).mark_rect().encode(
            x=alt.X('num_appts:O', title='Number of Appointments'),
            y=alt.Y('num_noshow:O', title='Number of No-shows', sort='descending'),
            color=alt.Color('num_patients:Q')
        ).properties(title='Heat map of appointments and no-show distribution')

    return chart


def plot_appt_noshow_tree_map(df):
    """
    Plot a treemap (https://plotly.com/python/treemaps/) showing the share of patients in each category of:
    number appointments & number of no shows. i.e. share of patients with 1 appointment and 0 no shows, share of
    patients with 1 appt, 1 no show, etc.

    Args:
        df: dataframe containing appointments with patient IDs and no-show information

    Returns:
        Treemap figure as described above
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
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))  # noqa

    return fig


def plot_numerical_feature_correlations(feature_df: pd.DataFrame) -> (alt.Chart, pd.DataFrame):
    """
    Calculates the correlation between the numerical features in the provided feature set (excluding the id columns,
    the NoShow column, and the columns which are sin & cos transformations of the temporal variables (if applicable).
    These latter columns are dropped since they will naturally be correlated with their original variables which are
    contained in the same feature set.

    These correlation values are then plotted in a histogram to visualise the amount of correlation between our features

    Args:
        feature_df: dataframe of features

    Returns:
        Histogram of correlation values and a dataframe containing these values
    """
    df_copy = feature_df.copy()
    df_copy = df_copy.drop(['FillerOrderNo', 'MRNCmpdId', 'NoShow'], axis=1)
    df_copy = df_copy.select_dtypes(include=[np.number])

    # remove columns which are sin & cos transformations of hour, month, time columns
    cols_to_drop = df_copy.columns[df_copy.columns.str.contains('_cos|_sin')]
    df_copy = df_copy.drop(cols_to_drop, axis=1)

    correlation_list = df_copy.corr().stack().reset_index().drop_duplicates()
    correlation_list.columns = ['var1', 'var2', 'correlation']
    correlation_list = correlation_list[correlation_list['var1'] != correlation_list['var2']]  # remove self-correlation

    # Remove duplicated rows (i.e. the correlation list at this stage will include correlation between var1 and var2,
    # as well as the correlation between var2 and var1 (the same value) so we keep just one of these
    def order_and_combine_cols(row):
        if row['var1'] < row['var2']:
            return row['var1'] + ", " + row['var2']
        else:
            return row['var2'] + ", " + row['var1']

    correlation_list['varcomb'] = correlation_list.apply(order_and_combine_cols, axis=1)
    correlation_list = correlation_list[['correlation', 'varcomb']].drop_duplicates()

    correlation_list['zero'] = 0  # Required for drawing vertical line at x=0 in altair chart

    chart = alt.Chart(correlation_list).mark_bar().encode(
        alt.X("correlation:Q", bin=True, title='Correlation measure'),
        alt.Y('count()', title='Number of feature pairs')
    )
    line = alt.Chart(correlation_list).mark_rule(color='black').encode(
        alt.X('zero')
    )
    return_chart = (chart + line).properties(title='Distribution of correlations between feature pairs')

    correlation_list = correlation_list.sort_values('varcomb')
    correlation_list = correlation_list.drop('zero', axis=1)

    return return_chart, correlation_list
