from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels import stats, tools
from nilearn import glm, interfaces, plotting


def f(row):
    """ """
    if not row['error'] and row['condition'] == 'unseen':
        cond = 'correct_rej'
    elif not row['error'] and row['condition'] == 'seen':
        cond = 'hit'
    elif row['error'] is True and row['condition'] == 'unseen':
        cond = 'false_alarm'
    elif row['error'] is True and row['condition'] == 'seen':
        cond = 'miss'
    else:
        cond = pd.NA
    return cond

t_r = 1.49
n_scans = 190
frame_times = (
    np.arange(n_scans) * t_r
)

events = sorted(Path(
    'things.fmriprep', 'sourcedata', 'things', 'sub-01'
).rglob('*events.tsv'))
event = events[0]
df = pd.read_csv(event, sep='\t')

df['memory_cond'] = df.apply(f, axis=1)
memory_events = pd.DataFrame(
    {
        "trial_type": df.memory_cond,
        "onset": df.onset_flip,
        "duration": (df.offset_flip - df.onset_flip)
    }
)

img_files = sorted(Path(
    'things.fmriprep', 'sub-01'
).rglob('*space-T1w_desc-preproc_part-mag_bold.nii.gz'))
confounds, _ = interfaces.fmriprep.load_confounds_strategy(
    str(img_files[0]), denoise_strategy=['compcor']
)

X1 = glm.first_level.make_first_level_design_matrix(
    frame_times=frame_times,
    events=memory_events,
    drift_model="polynomial",
    drift_order=3,
    # add_regs=motion,
    # add_reg_names=add_reg_names,
    hrf_model="glover",
)
plotting.plot_design_matrix(X1)
plt.show()


# from https://stackoverflow.com/a/48819434
X = tools.add_constant(X1)
pd.Series([stats.outliers_influence.variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)