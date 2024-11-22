import os
import itertools
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path, PurePath

from nilearn.plotting import plot_stat_map
from nilearn.glm.first_level import design_matrix, FirstLevelModel
from nilearn.image import high_variance_confounds, resample_img
from nilearn._utils import check_niimg

from ibc_public.utils_contrasts import make_contrasts
from ibc_public.utils_paradigm import make_paradigm, post_process
from ibc_public.utils_pipeline import fixed_effects_img, _load_summary_stats

subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-06']
tasks = ['emotion', 'gambling', 'language',
         'motor', 'wm', 'relational', 'social']
func_str = 'space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
regr_str = 'desc-confounds_timeseries.tsv'
gm_mask = ('./tpl-MNI152NLin2009cAsym_res-3mm_label-GM_'
           'desc-thr02_probseg.nii.gz')


def _subset_confounds(tsv, keep_confounds=None):
    """
    Only retain those confounds listed in `keep_confounds`

    Parameters
    ----------
    tsv : str
        Local file path to the fMRIPrep generated confound files
    keep_confounds : list
        A list of confounds to keep from the fMRIPrep generated files.

    Returns
    -------
    selected_confounds : np.recarray
    """
    if keep_confounds is None:
        keep_confounds = ['trans_x', 'trans_y', 'trans_z',
                          'rot_x', 'rot_y', 'rot_z']
    else:
        keep_confounds = keep_confounds

    # load in tsv and subset to only include our desired regressors
    tsv = str(tsv)
    confounds = np.recfromcsv(tsv, delimiter='\t', encoding='utf-8')

    try:
        confounds = confounds[keep_confounds]
    except ValueError:
        err_msg = ('Unrecognized confound requested.'
                   'Please confirm `keep_confounds` parameter.')
        raise(err_msg)

    names = list(np.lib.recfunctions.get_names(confounds.dtype))
    confounds = np.lib.recfunctions.structured_to_unstructured(confounds)

    return confounds, names


def generate_hcp_contrasts(task_id, event, scan, regr, mask_img,
                           tr=1.49, compcorr=True):
    """
    Parameters
    ----------
    task_id : str
        Task name
    events : str
        The path to the BIDS-formatted events.tsv file on disk
    scan : str
        The path to the fMRI scan on disk
    regr : str
        The path to the fMRIPrep-formatted regressors_timeseries.tsv
        file on disk
    mask_img : str
        The path to the grey matter mask on disk
    tr : float
        The time-to-repetition for the supplied fMRI scan. Default 1.49.
    compcorr : bool
        Whether or not to generate compcorr regressors, as created with
        nilearn.image.high_variance_confounds. Default True.

    Returns
    -------
    design : pd.DataFrame
        A generated design matrix, including stimulus events and motion
        regressors.
    contrasts : dict
        A dictionary of contrasts to compute, as defined in
        ibc_public.utils_contrasts.make_contrasts.
    """

    scan = check_niimg(scan)
    motion, motion_names = _subset_confounds(regr)

    if compcorr:
        confounds = high_variance_confounds(scan, mask_img=mask_img)
        confounds = np.hstack((confounds, motion))
        confound_names = ['conf_%d' % i for i in range(5)] + motion_names
    else:
        confounds = motion
        confound_names = motion_names

    if task_id == 'wm':
        df = pd.read_csv(event, index_col=None, sep='\t')
        df['trial_type'] = df['trial_type'] + df['stim_type']
        paradigm = post_process(df, 'wm')
    elif task_id == 'social':
        df = pd.read_csv(event, index_col=None, sep='\t')
        df['trial_type'] = df['trial_type'].str.title()
        paradigm = post_process(df, 'social')
    elif task_id == 'language':
        df = pd.read_csv(event, index_col=None, sep='\t')
        df['trial_type'] = df['trial_type'].str.rsplit('_', expand=True)[1]
        paradigm = post_process(df, 'hcp_language')
    elif task_id == 'motor':
        df = pd.read_csv(event, index_col=None, sep='\t')
        paradigm = post_process(df, 'hcp_motor')
    elif task_id == 'emotion':
        df = pd.read_csv(event, index_col=None, sep='\t')
        paradigm = post_process(df, 'emotional')
    else:  # if task in gambling, relational
        paradigm = make_paradigm(str(event), task_id)

    n_scans = scan.shape[-1]
    frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)

    design = design_matrix.make_first_level_design_matrix(
        frametimes, paradigm,
        add_regs=confounds, add_reg_names=confound_names)
    _, dmtx, names = design_matrix.check_design_matrix(design)

    contrasts = make_contrasts(f'hcp_{task_id}', names)

    return design, contrasts


def run_hcp_glm(scan, hcp_contrasts, mask_img, design_matrix,
                tr=1.49, smooth=None, subject_session_output_dir='./'):
    """
    Parameters
    ----------
    scan : str
        The path to the fMRI scan on disk
    hcp_contrasts : dict
        The generated dictionary of HCP contrasts to perform
    mask_img : str
        The path to the grey matter mask on disk
    design_matrix : pd.DataFrame
        A generated design matrix as supplied by generate_hcp_contrasts
    tr : float
        The time-to-repetition for the supplied fMRI scan. Default 1.49.
    subject_dic : dict
        A subject-specific dictionary. Default None (not used).
    smooth : int
        The FWHM of the Gaussian smoothing kernel to apply. Default None.
    subject_session_output_dir : str
        The path on disk to where to save the subject-specific outputs.
        Default is the current working directory.

    Returns
    -------
    z_maps :
    fmri_glm : nilearn.glm.first_level.FirstLevelModel object
        A fitted first-level model object, includes the
        computed design matrix(ces).
    """

    scan = check_niimg(scan)

    # GLM analysis
    print('Fitting a GLM (this takes time)...')
    fmri_glm = FirstLevelModel(mask_img=mask_img, t_r=tr, slice_time_ref=.5,
                               smoothing_fwhm=smooth).fit(
        scan, design_matrices=design_matrix)

    # compute contrasts
    z_maps = {}
    for contrast_id, contrast_val in hcp_contrasts.items():
        print(f"\tcontrast id: {contrast_id}")

        # store stat maps to disk
        for map_type in ['z_score', 'stat', 'effect_size', 'effect_variance']:
            stat_map = fmri_glm.compute_contrast(
                contrast_val, output_type=map_type)
            map_dir = os.path.join(
                subject_session_output_dir, f'{map_type}_maps')
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            map_path = os.path.join(map_dir, f'{contrast_id}.nii.gz')
            print(f"\t\tWriting {map_path} ...")
            stat_map.to_filename(map_path)

            # collect zmaps for contrasts we're interested in
            if map_type == 'z_score':
                z_maps[contrast_id] = map_path

    # stats_report_filename = os.path.join(
    #     subject_session_output_dir, 'report_stats.html')
    # report = make_glm_report(fmri_glm,
    #                          hcp_contrasts,
    #                          threshold=3.0,
    #                          cluster_threshold=15,
    #                          title=f'GLM for subject {sess_id}')
    # report.save_as_html(stats_report_filename)

    return z_maps, fmri_glm


def fixed_effects_analysis(hcp_contrasts, session_listing, mask_img,
                           task_id, subject_output_dir='./'):
    """
    Generates a fixed-effects analysis across sessions for each contrast
    in a given task.

    Parameters
    ----------
    hcp_contrasts : dict
        The generated dictionary of HCP contrasts to perform
    session_listing : list
        The list of session across which to generated fixed_effects
    mask_img : str
        The path to the grey matter mask on disk
    task_id : str
        The HCPTRT task for which to generate fixed effects maps
    subject_output_dir : str
        The path on disk to where to save the subject-specific outputs.
        Default is the current working directory.
    """
    write_dir = os.path.join(subject_output_dir,
                             f'res_stats_{task_id}_ffx')
    dirs = [os.path.join(write_dir, stat) for stat in [
        'effect_size_maps', 'effect_variance_maps', 'stat_maps']]

    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)
    print(write_dir)

    # iterate across contrasts
    for contrast in hcp_contrasts:
        print(f'fixed effects for contrast {contrast}. ')
        effect_size_maps, effect_variance_maps, data_available =\
            _load_summary_stats(
                subject_output_dir, session_listing, contrast)
        shape = nib.load(effect_size_maps[0]).shape
        if len(shape) > 3:
            if shape[3] > 1:  # F contrast, skipping
                continue
        ffx_effects, ffx_variance, ffx_stat = fixed_effects_img(
            effect_size_maps, effect_variance_maps, mask_img)
        nib.save(ffx_effects, os.path.join(
            write_dir, f'effect_size_maps/{contrast}.nii.gz'))
        nib.save(ffx_variance, os.path.join(
            write_dir, f'effect_variance_maps/{contrast}.nii.gz'))
        nib.save(ffx_stat, os.path.join(
            write_dir, f'stat_maps/{contrast}.nii.gz'))
        plot_stat_map(
            ffx_stat, display_mode='z',
            dim=0, cut_coords=7, title=contrast, threshold=3.0,
            output_file=os.path.join(
                write_dir, f'stat_maps/{contrast}.png'))


def run_subject_level(subject, task_id):
    """
    Generates session-level maps for a subject-task pairing, using
    fMRIPrep preprocessed outputs. Performs a fixed-effects analysis
    on these subject-level maps.

    Outputs can be optionally carried onto a group-level analysis.

    Parameters
    ----------
    subject : str
    task_id : str
    """
    scans = sorted(Path(
         'derivatives', 'fmriprep-20.2lts', 'fmriprep', subject).rglob(
         f'*_task-{task}*{func_str}'))
    regressors = sorted(Path(
          'derivatives', 'fmriprep-20.2lts', 'fmriprep', subject).rglob(
          f'*_task-{task}*{regr_str}'))
    events = sorted(Path(subject).rglob(
           f'*_task-{task}*events.tsv'))

    if len(scans) != len(events):
        err_msg = ("Number of event files does not match the "
                   "number of provided BOLD files. Please confirm "
                   "directory structure.")
        raise ValueError(err_msg)

    for func, regr, event in zip(scans, regressors, events):
        sess_id = event.name.split('_')[1]
        subject_session_output_dir = PurePath(subject,
                                              f'res_stats_{sess_id}')

        if not os.path.exists(subject_session_output_dir):
            os.makedirs(subject_session_output_dir)

        # NOTE : high_variance_confounds() is calling masking.apply_mask(),
        # which does NOT resample the mask or image, so we run into an
        # affine error (the affine of the mask is 0.5mm off from the
        # affine of the images)
        if gm_mask is not None:
            aff_orig = nib.load(gm_mask).affine[:, -1]
        else:
            aff_orig = nib.load(func).affine[:, -1]
        target_affine = np.column_stack([np.eye(4, 3) * 3, aff_orig])
        scan = resample_img(img=str(func),
                            target_affine=target_affine)

        design, hcp_contrasts = generate_hcp_contrasts(
            task, event, scan, regr, mask_img=gm_mask)
        np.savez(PurePath(subject_session_output_dir, 'design_matrix.npz'),
                 design_matrix=design)

        z_maps, fmri_glm = run_hcp_glm(
            scan, hcp_contrasts, gm_mask, design,
            subject_session_output_dir=subject_session_output_dir)

    session_listing = np.unique([e.name.split('_')[1] for e in events])

    fixed_effects_analysis(
        hcp_contrasts, session_listing, mask_img=gm_mask, task_id=task,
        subject_output_dir=subject_session_output_dir.parent)


if __name__ == "__main__":

    for subject, task in itertools.product(subjects, tasks):

        run_subject_level(subject, task)
        print(f"Finished running task {task} for subject {subject}")