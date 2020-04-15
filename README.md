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

# build row-per-slot (no show or completed) data set
slot_df = mridle.data_management.build_slot_df(status_df)
```


### Integrate DICOM Data
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
Compare to examples collected manually from Dispo
(only works on USZ machine)
```python
dispo_examples = dm['dispo_data']['manual_examples.yaml'].load()
dispo_df = mridle.data_management.build_dispo_df(dispo_examples)

day = 14
month = 1
year = 2019
machine = 'MR1'
slot_status = 'show'
dispo_patids, slot_df_patids = mridle.data_management.validate_against_dispo_data(dispo_df, slot_df, day, month, year, slot_status)
```

## Tests
`mridle` contains a test suite for validating the no-show identification algorithm.
Run the tests by navigating to the `mridle` directory and running `pytest`.
