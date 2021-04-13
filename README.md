# MRIdle
Resource Optimization for Radiology

## Example Usage

### Load Data
```python
%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from IPython.display import display

from datatc import DataManager
import mridle
from mridle.data_management import SHOW_COLS


dm = DataManager('mridle')
raw_df = dm['rdsc_extracts']['five_years'].select('parquet').load()
# load patient ids that are known to be test patient ids (not real patients)
test_pat_ids = dm['dispo_data']['exclude_patient_ids.yaml'].load()

# build row-per-status-change data set
status_df = mridle.data_management.build_status_df(raw_df, exclude_pat_ids=test_pat_ids)

# build row-per-slot (no show slot or completed slot) data set
slot_df = mridle.data_management.build_slot_df(status_df)
```

where `status_df` contains the columns:
| column name | type | description |
|---|---|---|
| FillerOrderNo | int | appointment id |
| MRNCmpdId | object | patient id |
| date | datetime | the date and time of the status change |
| was_status | category | the status the appt changed from |
| now_status | category | the status the appt changed to |
| was_sched_for | float | number of days ahead the appt was sched for before status change relative to `date` |
| now_sched_for | int | number of days ahead the appt is sched for after status change relative to `date` |
| was_sched_for_date | datetime | the date the appt was sched for before status change |
| now_sched_for_date | datetime| the date the appt is sched for after status change |
| patient_class_adj | object | patient class (adjusted) ['ambulant', 'inpatient'] |
| NoShow | bool | [True, False] |
| NoShow_severity | object | ['hard', 'soft'] |
| slot_outcome | object | ['show', 'rescheduled', 'canceled'] |
| slot_type | object | ['no-show', 'show', 'inpatient'] |
| slot_type_detailed | object | ['hard no-show', 'soft no-show', 'show', 'inpatient'] |

and `slot_df` contains the columns:
| column name | type | description |
|---|---|---|
|FillerOrderNo | int | appointment id |
| MRNCmpdId | object | patient id |
| start_time | datetime | appt scheduled start time |
| end_time | datetime | appt scheduled end time |
| NoShow | bool | [True, False] |
| slot_outcome | object | ['show', 'rescheduled', 'canceled'] |
| slot_type | object | ['no-show', 'show', 'inpatient'] |
| slot_type_detailed | object | ['hard no-show', 'soft no-show', 'show', 'inpatient'] |
| EnteringOrganisationDeviceID | object | device the appt was scheduled for |
| UniversalServiceName | object | the kind of appointment |


### Integrate DICOM Data
The appointment start times, end times, and radiology device used as described by the appointment status change data are unreliable.
Integrate the status change dataset with DICOM data to get accurate start time, end time, and device usage information.
```python
import sqlite3

dicom_db_path = dm['dicom_data']['2020_02_09_dicom_from_Christian'].select('sqlite').path
query_text = dm['dicom_data']['2020_02_09_dicom_from_Christian'].select('image_times.sql').load(data_interface_hint='txt')
c = sqlite3.connect(dicom_db_path)
dicom_times_df = pd.read_sql_query(query_text, c)
dicom_times_df = mridle.data_management.format_dicom_times_df(dicom_times_df)
slot_w_dicom_df = mridle.data_management.integrate_dicom_data(slot_df, dicom_times_df)

```

### Look at Example Appointments

To look at an example appointment history:
```python
fon = 5758396
appt = mridle.exploration_utilities.view_status_changes(status_df, fon)
display(appt[SHOW_COLS])
```


To look at a random example No Show appointment:
```python
for i in range(50):
    appt = mridle.exploration_utilities.view_status_changes_of_random_sample(status_df)

    if appt['NoShow'].max() == 0:
        continue
    else:
        display(appt[SHOW_COLS])
        break
```

### Plotting

`altair` plotting

```python
import altair as alt

alt.renderers.enable('default')


# the altair plot needs no-show end times set (by default they're NAT)
slot_df['end_time'] = slot_df.apply(mridle.data_management.set_no_show_end_times, axis=1)

mridle.plotting_utilities.alt_plot_date_range_for_device(slot_df, 'MR1', end_date='04/17/2019')

# you can also highlight just one kind of appointment
mridle.plotting_utilities.alt_plot_date_range_for_device(slot_df, 'MR1', end_date='04/17/2019', highlight='no-show')
```

`matplotlib` plotting

Plot a day:
```python
%matplotlib inline

year = 2019
month = 1
day = 14

mridle.plotting_utilities.plot_a_day(slot_df, year, month, day, labels=False, alpha=0.5)
```

Plot a day for one device:
```python
mridle.plotting_utilities.plot_a_day_for_device(slot_df, 'MR-N1', year, month, day, labels=True, alpha=0.5)
```

### Data Validation

#### Plot validation experiment.

This function generates a scatter bar plot, where each scatter-bar corresponds to a single year, and every point within
that scatter-bar corresponds to a single day. Each point representing a single day is obtained by aggregating
appointment data from that day. It is calculated by taking the ratio of appointments in the extract divided by the
number of appointments in the dispo data. Ideally each one of these points should be close to 1 since the numbers
should be similar. A value over 1 represents a larger number of appointments in the extract than viewed by the dispo.
(only works on USZ machine)

```python
dispo_data_1a = dm['dispo_data']['experiment1A.yaml'].load()
dispo_data_1b = dm['dispo_data']['experiment1B.yaml'].load()
test_pat_ids = dm['dispo_data']['exclude_patient_ids.yaml'].load()
dispo_data_e1 = dispo_data_1a + dispo_data_1b
dispo_e1_df = mridle.data_management.build_dispo_exp_1_df(dispo_data_e1, exclude_patient_ids=test_pat_ids)

dispo_data_2 = dm['dispo_data']['experiment2.yaml'].load()
dispo_data_2_corrections = dm['dispo_data']['experiment2_corrections.yaml'].load()
dispo_e2_records = dispo_data_2 + dispo_data_2_corrections
dispo_e2_slot_df = mridle.data_management.build_dispo_exp_2_df(dispo_e2_records)
```

```python
slot_type_detailed = 'show'
df_exp1 = mridle.data_management.generate_data_firstexperiment_plot(dispo_df, slot_df)
df_ratios = mridle.data_management.calculate_ratios_experiment(df_exp1, slot_type_detailed)
mridle.plotting_utilities.plot_validation_experiment(df_ratios)
```

#### Plot scatter bar jaccard per type

To compare data from the extract to the examples collected manually from Dispo. Now using the Jaccard Index.

This function generates a scatter bar plot, where each scatter-bar corresponds to a single year, and every point within
that scatter-bar corresponds to a single day. Each point representing a single day is obtained by analyzing appointment
data from that day. It is calculated by calculated the Jaccard Index.  When the sets match completely this number is 1.

```python
from mridle.plotting_utilities import plot_scatter_bar_jaccard_per_type

dispo_examples = dm['dispo_data']['manual_examples.yaml'].load()
dispo_df = mridle.data_management.build_dispo_df(dispo_examples)
slot_type_detailed = 'show'

plot_scatter_bar_jaccard_per_type(dispo_df, slot_df, slot_type_detailed)
```

#### Plot dispo extract slot diffs 

This function compares data from the extract to the examples collected manually from the dispo by generating a scatter plot. X axis plots the number of appointments that are in the dispo but not in the hospital extract. Y axis plots the number of appointments that are in the hospital extract but not in the dispo. The closer that the points are plotted close to (0,0), the better the matching between Dispo and the hospital extract is.

```python
from mridle.plotting_utilities import plot_dispo_extract_slot_diffs

dispo_examples = dm['dispo_data']['manual_examples.yaml'].load()
dispo_df = mridle.data_management.build_dispo_df(dispo_examples)
slot_type_detailed = 'show'

plot_dispo_extract_slot_diffs(dispo_df, slot_df, slot_type_detailed)
```

#### Plot scatter dispo extract slot cnt

This function compares data from the extract to the examples collected manually from Dispo. It generates a scatter plot where all available data is collected and split by 'slot_type_detailed' ('show', 'soft no-show', 'hard no-show'). X axis plots the number of appointments in the dispo whereas Y axis plots the number of appointments in the hospital extract. Ideally, the plotted points should remain in the diagonal.

```python
from mridle.plotting_utilities import plot_scatter_dispo_extract_slot_cnt

dispo_examples = dm['dispo_data']['manual_examples.yaml'].load()
dispo_df = mridle.data_management.build_dispo_df(dispo_examples)

plot_scatter_dispo_extract_slot_cnt(dispo_df, slot_df)
```

#### Plot scatter dispo extract slot cnt for type

To compare data from the extract to the examples collected manually from Dispo. Still with scatter plots.

This function compares data from the extract to the examples collected manually from Dispo. It generates a scatter plot where all available data is collected for a given 'slot_type_detailed' ('show', 'soft no-show', 'hard no-show'). Different colors in the graph represent different years collected in the data.number X axis plots the number of appointments in the dispo whereas Y axis plots the number of appointments in the hospital extract. Ideally, the plotted points should remain in the diagonal.

```python
from mridle.plotting_utilities import plot_scatter_dispo_extract_slot_cnt_for_type

dispo_examples = dm['dispo_data']['manual_examples.yaml'].load()
dispo_df = mridle.data_management.build_dispo_df(dispo_examples)
slot_type_detailed = 'hard no-show'

plot_scatter_dispo_extract_slot_cnt_for_type(dispo_df, slot_df, slot_type_detailed)
```

#### Plot idle time figures

There are multiple ways that we use to visualise the historic idle time for each machine. The first of these gives a bar graph with each horizontal row indicating one day, with time on the x-axis. These bars (representing days) are then coloured according to the state (active, idle, or 'buffer' time) the MRI machine was in at that given time, giving an overview of the past scheduling of the machines. Code to generate this plot is outlined below (assuming dicom data has already been obtained and processed).

```python
from mridle.idle_time import calc_idle_time_gaps, calc_appts_and_gaps, plot_daily_appt_idle_segments

idle_df = calc_idle_time_gaps(dicom_times_df, time_buffer_mins=5)
appts_and_gaps = calc_appts_and_gaps(idle_df)
plot_daily_appt_idle_segments(appts_and_gaps, width=500, height=350)
```
The second plot type which we show here now gives a vertical bar for each day, with the y-axis now representing total hours (or as a percentage of the whole day). Again, these bars are coloured according to the total time the MRI machine spent in each of the three states during the day. Generation of this plot is outlined below (assuming dicom data has already been obtained and processed).


```python
from mridle.idle_time import calc_idle_time_gaps, calc_daily_idle_time_stats, plot_total_active_idle_buffer_time_per_day

idle_df = calc_idle_time_gaps(dicom_times_df, time_buffer_mins=5)
daily_idle_stats = calc_daily_idle_time_stats(idle_df)
plot_total_active_idle_buffer_time_per_day(reasonable_hours)
```

#### Data Validation Experiment 2: Rescheduled NoShows
```python
exp_2 = dm['dispo_data'].select('2').load()
test_pat_ids = dm['dispo_data']['exclude_patient_ids.yaml'].load()
dispo_e2_df = mridle.data_management.build_dispo_df(exp_2, exclude_patient_ids=test_pat_ids)
dispo_e2_df = mridle.data_management.find_no_shows_from_dispo_exp_two(dispo_e2_df)

# build rdsc dataframe to compare to
rdsc_exp_2_df = dm['rdsc_extracts'].select('exp_2').select('RIS_2020_week40_fix_column_headers.csv').load()
rdsc_exp_2_status_df = mridle.data_management.build_status_df(rdsc_exp_2_df, exclude_patient_ids=test_pat_ids)
rdsc_exp_2_slot_df = mridle.data_management.build_slot_df(rdsc_exp_2_status_df)

# plot daily Jaccard scores
mridle.plotting_utilities.plot_scatter_bar_jaccard_per_type(dispo_e2_df, rdsc_exp_2_slot_df, 'rescheduled')
```

## Constructing Model Feature Sets
Feature sets are constructed from `status_df`, using functionality from `mridle.feature_engineering`.

Here's an example of using `datatc` to create and save a feature set:
```python
dm['feature_sets'].save(status_df, 'harvey_seven.csv', mridle.feature_engineering.build_harvey_et_al_features_set)
```
This uses `datatc`'s Saved Data Transform functionality to save a dataset and the code that generated it.
This line of code:
  * consumes `status_df`
  * applies `mridle.feature_engineering.build_harvey_et_al_features_set`
  * saves the result as `harvey_seven.csv`
  * also stamps the code contained in `build_harvey_et_al_features_set` alongside the dataset for easy future reference

Using `datatc`, one can not only load the dataset, but also view and re-use the code that generated that dataset:
```python
sdt = dm['feature_sets'].latest().load()

# view and use the data
sdt.data.head()

# view the code that generated the dataset
sdt.view_code()

```

## Modeling
```python
from datatc import DataManager
from mridle.experiment import ModelRun, PartitionedExperiment
from mridle.experiments.harvey import HarveyModel
from sklearn.ensemble import RandomForestClassifier

# load the latest feature set
dm = DataManager('mridle')
sdt = dm['feature_sets'].latest().load()

# specify parameters of the model
name='your human readable experiment name here'
label_key = 'noshow'
model = RandomForestClassifier()
hyperparams = {'n_estimators': [10, 100, 500]}
n_partitions = 5

# PartitionedExperiment runs your modeling experiemnt <n_partition> times, on separate test set partitions.
# By default, the partitions are stratified by label. 
exp = PartitionedExperiment(name=name, data_set=sdt.data, label_key=label_key, preprocessing_func=sdt.func,
                            model_run_class=HarveyModel, model=model, hyperparams=hyperparams,
                            n_partitions=n_partitions, verbose=True)
results = exp.run()
print(results)
print("Evaluation")
print(exp.show_evaluation())
print("Feature Importances")
print(exp.show_feature_importances())

# you can inspect each of the ModelRun objects
mr_0 = exp.model_runs['Partition 0']
```

## Tests
`mridle` contains a test suite for validating the no-show identification algorithm.
Run the tests by navigating to the `mridle` directory and running `pytest`.
