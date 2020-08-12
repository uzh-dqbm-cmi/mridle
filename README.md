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
raw_df = dm['rdsc_extracts']['2020-02-04_RIS_deID_3months'].select('xlsx').load()

# build row-per-status-change data set
status_df = mridle.data_management.build_status_df(raw_df)

# build row-per-slot (no show slot or completed slot) data set
slot_df = mridle.data_management.build_slot_df(status_df)
```

where `status_df` contains the columns:
| column name | type | description |
|---|---|---|
| FillerOrderNo | int | appt id |
| MRNCmpdId | int | patient id |
| date | datetime | the date and time of the status change |
| was_status | str | the status the appt changed from |
| now_status | str | the status the appt changed to |
| was_sched_for | int | number of days ahead the appt was sched for before status change relative to `date` |
| now_sched_for | int | number of days ahead the appt is sched for after status change relative to `date` |
| was_sched_for_date | datetime | the date the appt was sched for before status change |
| now_sched_for_date | datetime| the date the appt is sched for after status change |
| patient_class_adj | patient  |lass (adjusted) ['ambulant', 'inpatient'] |
| NoShow | bool | [True, False] |
| NoShow_severity | str | ['hard', 'soft'] |
| slot_outcome | str | ['show', 'rescheduled', 'canceled'] |
| slot_type | str | ['no-show', 'show', 'inpatient'] |
| slot_type_detailed | str | ['hard no-show', 'soft no-show', 'show', 'inpatient'] |

and `slot_df` contains the columns:
| column name | type | description |
|---|---|---|
|FillerOrderNo | int | appt id |
| MRNCmpdId | int | patient id |
| start_time | datetime | appt scheduled start time |
| end_time | datetime | appt scheduled end time |
| NoShow | bool | [True, False] |
| slot_outcome | str | ['show', 'rescheduled', 'canceled'] |
| slot_type | str | ['no-show', 'show', 'inpatient'] |
| slot_type_detailed | str | ['hard no-show', 'soft no-show', 'show', 'inpatient'] |
| EnteringOrganisationDeviceID | str | device the appt was scheduled for |
| UniversalServiceName | str | the kind of appointment |


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
dispo_examples = dm['dispo_data']['manual_examples.yaml'].load()
dispo_df = mridle.data_management.build_dispo_df(dispo_examples)

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

## Tests
`mridle` contains a test suite for validating the no-show identification algorithm.
Run the tests by navigating to the `mridle` directory and running `pytest`.
