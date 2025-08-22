# things-glm

Code for General Linear Model (GLM) analyses of memory contrasts in the CNeuroMod-THINGS dataset.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Fixed-effects analyses on preprocessed BOLD data

This code assumes that you have access to files in the `things.fmriprep`, `things`, `anat.pycortex` datasets; in particular:

- the preprocessed BOLD files in native space (`*space-T1w_desc-preproc_bold.nii.gz`)
- the corresponding functional masks (`*space-T1w_desc-brain_mask.nii.gz`)
- the associated confounds file (`*desc-confounds_timeseries.tsv`)
- the events files, describing the task (`*ses-???_*events.tsv`). NB: These should be with the three-digit session identifiers, indicating manual review by the data curator

The code is primarily based on the `gen-memory-ffx.py` script, which generates fixed-effects analysis (FFX) stat maps from BOLD data preprocessed with fmriprep.

There is an additional quality-check script, `check-memory-counts.py`, to assess the distribution of memory conditions
(e.g., "correct rejection") within individual scanning runs.

The `visualize-memory-ffx.py` script can also bee used to visualize stat maps (e.g., betas and t-scores) on anatomical volumes and on cortical flat maps. To run this script, you will need access to the pycortex filestore (database), which contains pycortex files (e.g., `matrices.xfm`) for all Courtois-NeuroMod participants.

Each script is designed to be run at the command-line with options visible via `--help`.

Note that visualizing flat maps requires you to have Inkscape available on your system.
On MacOSx, you will additionally need to add it to your Python PATH.
For more information,
please see the [pycortex documentation](https://gallantlab.org/pycortex/install.html) and `visualize-memory-ffx.py`.

On all systems, you will additionally need to configure `cortex.database.default_filestore`.
This can be set via modifying the `pycortex` configuration file,
accessible at the path provided by `cortex.options.usercfg` after installation.

## two-sampled t-tests on GLMsingle betas

The `glm-mem-betas.py` script performs two-sampled t-tests on trial-wise betas estimated with GLMsingle.

This code requires you to have access to files inside submodules in the `cneuromod-things` repository; in particular:
- normalized trial-wise betas (`cneuromod-things/THINGS/glmsingle/sub-*/glmsingle/output/sub-*_task-things_space-T1w_model-fitHrfGLMdenoiseRR_stat-trialBetas_desc-zscore_statseries.h5`)
- trial-wise behavioural metrics (`THINGS/behaviour/sub-*/beh/sub-*_task-things_desc-perTrial_annotation.tsv`)


The subject-specific folders included in this repository contain the outputs of `gen-memory-ffx.py` and `glm-mem-betas.py`,
though these can be re-generated (and additional outputs saved, including the run-level design matrices), if you have access to all of the data detailed above.
