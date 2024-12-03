from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import glm, image, plotting
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.contrasts import compute_fixed_effects
from nilearn.interfaces.fmriprep import load_confounds_strategy


def f(row):
    """
    # TODO : Double-check that these are being correctly generated
    """
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


img_files = sorted(Path(
    'things.fmriprep', 'sub-01'
).rglob('*space-T1w_desc-preproc_bold.nii.gz'))
# only grab events with three digit ses nums ; indicates corrected
events = sorted(Path(
    'things.fmriprep', 'sourcedata', 'things', 'sub-01'
).rglob('*ses-???_*events.tsv'))

# drop ses-01 / ses-001 from images, events
img_files = list(filter( lambda i: ('ses-01' not in str(i)), img_files))
events = list(filter( lambda e: ('ses-001' not in str(e)), events))

# TODO: load this automatically from associated image files
t_r = 1.49
n_scans = 190
frame_times = (
    np.arange(n_scans) * t_r
)

design_matrices = []
stats_imgs = []
for img, event in zip(img_files, events):

    confounds, _ = load_confounds_strategy(
        str(img),
        denoise_strategy='compcor',
        compcor='temporal_anat_combined',
        n_compcor=10,
    )

    # load in events files and create memory conditions
    # based on performance
    df = pd.read_csv(event, sep='\t')
    df['memory_cond'] = df.apply(f, axis=1)
    memory_events = pd.DataFrame(
        {
            "trial_type": df.memory_cond,
            "onset": df.onset,
            "duration": df.duration
        }
    )
    if memory_events.duplicated('onset').any():
        print(f'Detected duplicate events in {event}!')

    # generate design matrices
    # TODO: Select sensible choices here
    design_matrix = glm.first_level.make_first_level_design_matrix(
        frame_times=frame_times,
        events=memory_events,
        # drift_model="polynomial",
        # drift_order=3,
        add_regs=confounds,
        add_reg_names=confounds.columns,
        hrf_model="glover",
    )
    design_matrices.append(design_matrix)

    fmri_glm = FirstLevelModel(t_r=t_r, smoothing_fwhm=5)
    fmri_glm = fmri_glm.fit(img, design_matrices=design_matrix)

    contrast_val = (design_matrix.columns == 'hit') * 1.0 -\
                        (design_matrix.columns == 'correct_rej')
    stats_img = fmri_glm.compute_contrast(contrast_val, output_type='all')
    stats_imgs.append(stats_img)

# for design_matrix in design_matrices:    
#     plotting.plot_design_matrix(design_matrix)
#     plt.show()

fixed_fx_contrast, fixed_fx_variance, fixed_fx_stat = compute_fixed_effects(
    [simg['effect_size'] for simg in stats_imgs],
    [simg['effect_variance'] for simg in stats_imgs]
)
plotting.plot_stat_map(
    fixed_fx_stat,
    bg_img=image.mean_img(img_files[0]),
    threshold=3.0,
    display_mode="z",
    cut_coords=3,
    black_bg=True,
    title="hit-correct_rej",
)
plt.show()

# from https://stackoverflow.com/a/48819434
# X = tools.add_constant(X1)
# pd.Series([stats.outliers_influence.variance_inflation_factor(X1.values, i) 
#                for i in range(X1.shape[1])], 
#               index=X1.columns)