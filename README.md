# MRIdle
Resource Optimization for Radiology

## Example Usage
```python
%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt

from datatc import DataManager
import mridle
from mridle.data_management import SHOW_COLS

%matplotlib inline


dm = DataManager('mridle')
raw_df = dm['rdsc_extracts']['2020-02-04_RIS_deID_3months'].select('xlsx').load()

# build row-per-status-change data set
df = mridle.data_management.build_status_df(raw_df)

# build row-per-slot (no show or completed) data set
one_per_slot = mridle.data_management.build_slot_df(df)


```
To look at an example appointment history:
```python
fon = 5758396
appt = mridle.exploration_utilities.view_status_changes(df, fon)
display(appt[SHOW_COLS])
```


To look at a random example No Show appointment:
```python
for i in range(50):
    appt = mridle.exploration_utilities.view_status_changes_of_random_sample(df)

    if appt['NoShow'].max() == 0:
        continue
    else:
        display(appt[SHOW_COLS])
        break
```

Plot a day:
```python
year = 2019
month = 1
day = 14

mridle.plotting_utilities.plot_a_day(one_per_slot, year, month, day, labels=False, alpha=0.5)
```

Plot a day for one device:
```python
mridle.plotting_utilities.plot_a_day_for_device(one_per_slot, 'MR-N1', year, month, day, labels=True, alpha=0.5)
```