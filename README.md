# MRIdle
Resource Optimization for Radiology

## Getting started

### Setup
MRIdle deals with patient data, therefore we work on a dedicated machine ("Louisa") which is managed by USZ.

#### Access the "Louisa" computing environment
1. If you don't have a regular USZ account, get one now.
2. Get your account on Louisa: See Notion for instructions for how to access Louisa.
3. Log onto Windows on an USZ machine or via remote desktop (mypc.usz.ch).
4. Connect to Louisa through SSH: Open PuTTY (type "putty" in start menu), enter `Louisa's IP address` (see Notion page) as the host name and press "open".
5. Now log in using your `ACC` account information.
6. Optional: you can now set a new password on this linux machine with the command `passwd`.

#### Installation
1.  Install Miniconda:
    ```
    cp /tmp/Miniconda3-latest-Linux-x86_64.sh .
    bash Miniconda3-latest-Linux-x86_64.sh
    ```
1. Create a MRIdle python environment:
    ```
    conda create --name mridle python=3.8
    ```
1. Activate the environment:
    ```
    conda activate mridle
    ```
1. `git clone` the MRIdle repo into your home directory via HTTPS:
    ```
    git clone https://github.com/uzh-dqbm-cmi/mridle.git
    ```
   Note: you will have to use GitHub HTTPS authentication with a [Personal Access Token](https://docs.github.com/en/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token)
1. Move into the mridle directory:
    ```
    cd mridle
    ```
1. Install the package and it's requirements:
    ```
    pip install -r src/requirements.txt
    ```

#### Set Up Jupyter

1. Ask someone in the team to assign you a port for running Jupyter notebooks.
1. Connect your dedicated port to your localhost:8888 port using `ssh -N -L localhost:8888:localhost:your-port your-acc-username@louisa-ip-address` in the Windows command line `cmd`. Alternatively save this command in a `.bat` file.
1. Start Jupyter lab through kedro in order to access kedro functionality:
    ```
    kedro jupyter lab /data/mridle/
    ```
    Note: you must run this command from the top level `mridle` repo directory. 
1. In your browser, go to `localhost:8888` to open Jupyter.
1. In a notebook, run the following code to import the mridle module. This code snippet also activates the autoreload IPython magic so that the module automatically updates with any code changes you make.
    ```
    %load_ext autoreload
    %autoreload 2

    import mridle
    ```

### Rules
- Do not delete anything.
- Patient data, even anonymised, always stays on Lousia.
- Naming convention for jupyter notebooks: `number_your-initials_short-description`.

## MRIdle + Kedro
MRIdle uses [Kedro](https://kedro.readthedocs.io) for organizing the data pipelines.
Kedro is an open-source Python framework for creating reproducible, maintainable and modular data science code.
It borrows concepts from software engineering best-practice and applies them to machine-learning code.

### Project Structure
Here is a high level overview of this repo's kedro project structure (adapted from [this Kedro doc page](https://kedro.readthedocs.io/en/stable/12_faq/02_architecture_overview.html)):
* The `conf/` directory contains configuration for the project, including:
    * `base/catalog/*.yml` contain data catalog entries for all data files that are involved in the pipelines.
    * `base/parameters.yml` is where parameters for pipelines is stored, for example model training parameters.
* The `src/` directory contains the source code for the project, including:
    * `mridle/` is the package directory, and contains:
        * The `pipelines/` directory, which contains the source code for your pipelines.
        * The `utiltities/` directory contains source code that is shared across multiple pipelines, or is independent from pipelines.
        * `pipeline_registry.py` file defines the project pipelines, i.e. pipelines that can be run using kedro run --pipeline.
    * `tests/` is where the tests go
    * `requirements.in` contains the source requirements of the project.
* `pyproject.toml` identifies the project root by providing project metadata.

### Kedro Viz
Kedro organizes data transformation steps into pipelines.
The easiest way to explore the pipelines is via Kedro's visualization tool, which you can open by running `kedro viz` and opening the webapp in your browser.

Below is a short summary of some of the Kedro functionality you may use to work with MRIdle.
You can read much more in the [Kedro documentation](https://kedro.readthedocs.io)!

### Kedro on the command line

#### Running kedro pipelines

To run a pipeline on the command line, run
```
kedro run --pipeline "<pipeline name>"
```

You can also specify which nodes to start from or stop at:
```
kedro run --pipeline "<pipeline name>" --from-nodes "<nodename>"
```

### Using kedro in Jupyter & IPython
You can also interact with kedro via Jupyter and IPython sessions.
To start a Jupyter or IPython session with kedro activated, run `kedro jupyter lab /data/mridle/` or `kedro ipython` from within the `mridle` directory.
Running Jupyter and IPython via kedro grants you access to 3 kedro variables:
* `catalog`: Load data created by kedro pipelines
* `context`: Access information about the pipelines
* `session`: Run pipelines
(if at any point you want to refresh these variables with changes you've made, run `%reload_kedro`)

#### Kedro data catalog
The Kedro data catalog makes loading data files from the pipelines easy:
```
slot_df = catalog.load('slot_df')
```
With this method, you can load any file defined in the Data Catalog defined in `conf/base/catalog.yml`

#### Running kedro pipelines
Here are some example commands for running pipelines within Jupyter/IPython:
```
session.run(pipeline_name='harvey')
session.run(ppipeline_name='harvey', from_nodes=['train_harvey_model')
```

## Example Usage

`status_df` contains the columns:
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

`slot_df` contains the columns:
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


### Look at Example Appointments

To look at an example appointment history:
```python
fon = 5758396
appt = mridle.utilities.exploration_utilities.view_status_changes(status_df, fon)
display(appt[SHOW_COLS])
```


To look at a random example No Show appointment:
```python
for i in range(50):
    appt = mridle.utilities.exploration_utilities.view_status_changes_of_random_sample(status_df)

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

mridle.utilities.plotting_utilities.alt_plot_date_range_for_device(slot_df, 'MR1', end_date='04/17/2019')

# you can also highlight just one kind of appointment
mridle.utilities.plotting_utilities.alt_plot_date_range_for_device(slot_df, 'MR1', end_date='04/17/2019', highlight='no-show')
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
mridle.utilities.plotting_utilities.plot_a_day_for_device(slot_df, 'MR-N1', year, month, day, labels=True, alpha=0.5)
```

## Tests
`mridle` contains a test suite for validating the data pipelines, including the no-show identification algorithm.
Run the tests by navigating to the top level `mridle` directory and running `kedro test`.
