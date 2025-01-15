# things-glm

Code for General Linear Model (GLM) analyses of memory contrasts in the THINGS dataset.

This code assumes that you have access to files in the `things.fmriprep` and and `things` datasets; in particular:
- the preprocessed BOLD files in native space (`*space-T1w_desc-preproc_bold.nii.gz`)
- the corresponding functional masks (`*space-T1w_desc-brain_mask.nii.gz`)
- the associated confounds file (`*desc-confounds_timeseries.tsv`)
- the events files, describing the task (`*ses-???_*events.tsv`). NB: These should be with the three-digit session identifiers, indicating manual review by the data curator

The code is primarily based on two scripts:
1. `gen-memory-ffx.py`, which will generate fixed-effects analysis (FFX) stat maps
1. `visualize-memory-ffx.py`, which will visualize these stat maps on the volume and on flat maps

There is an additional quality-check script,
`check-memory-counts.py` to assess the distribution of memory conditions
(e.g., "correct rejection") within individual scanning runs.

Each script is designed to be run at the command-line with options visible via `--help`.

Note that visualizing the flat maps requires you to have Inkscape available on your system.
On MacOSx, you will additionally need to add it to your Python PATH.
For more information,
please see the [pycortex documentation](https://gallantlab.org/pycortex/install.html) and `visualize-memory-ffx.py`.
