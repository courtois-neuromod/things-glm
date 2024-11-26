from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels import stats
import matplotlib.pyplot as plt
from nilearn import glm, image, plotting
from nilearn.glm.first_level import FirstLevelModel
from nilearn.interfaces.fmriprep import load_confounds_utils as utils
from nilearn.interfaces.fmriprep.load_confounds import _load_single_confounds_file


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
).rglob('*space-T1w_desc-preproc_part-mag_bold.nii.gz'))
events = sorted(Path(
    'things.fmriprep', 'sourcedata', 'things', 'sub-01'
).rglob('*events.tsv'))


# TODO: load this automatically from associated image files
t_r = 1.49
n_scans = 190
frame_times = (
    np.arange(n_scans) * t_r
)

design_matrices = []
for img, event in zip(img_files[:20], events[:20]):

    # get associated confounds file
    # can be streamlined once 
    # https://github.com/courtois-neuromod/cneuromod-things/issues/53
    # is resolved
    # TODO: Select sensible choices here
    confounds_file = utils._get_file_name(img)
    confounds_json = utils.get_json(confounds_file)
    _, confounds = _load_single_confounds_file(
        confounds_file,
        strategy=['motion', 'compcor'],
        demean=True,
        confounds_json_file=confounds_json,
        motion='basic',
        compcor='anat_combined',
        n_compcor=10,
    )

    # load in events files and create memory conditions
    # based on performance
    df = pd.read_csv(event, sep='\t')
    df['memory_cond'] = df.apply(f, axis=1)
    memory_events = pd.DataFrame(
        {
            "trial_type": df.memory_cond,
            "onset": df.onset_flip,
            "duration": (df.offset_flip - df.onset_flip)
        }
    )

    # generate design matrices
    # TODO: Select sensible choices here
    design_matrix = glm.first_level.make_first_level_design_matrix(
        frame_times=frame_times,
        events=memory_events,
        drift_model="polynomial",
        drift_order=3,
        add_regs=confounds,
        add_reg_names=confounds.columns,
        hrf_model="glover",
    )
    design_matrices.append(design_matrix)
    
# plotting.plot_design_matrix(design_matrix)
# plt.show()

fmri_glm = FirstLevelModel()
fmri_glm = fmri_glm.fit(img_files[:20], design_matrices=design_matrices)
contrast_val = (design_matrix.columns == 'hit') * 1.0 -\
                    (design_matrix.columns == 'correct_rej')

z_map = fmri_glm.compute_contrast(contrast_val, output_type='z_score')
plotting.plot_stat_map(
    z_map,
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