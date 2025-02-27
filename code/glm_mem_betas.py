import sys, glob

import argparse
from pathlib import Path

import h5py
import nibabel as nib
from nilearn.masking import unmask
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
#from scipy.stats import norm
import tqdm


def load_files(idir, sub):

    betas_path = Path(
        f"{idir}/THINGS/glmsingle/sub-{sub}/glmsingle/output/"
        f"sub-{sub}_task-things_space-T1w_model-fitHrfGLMdenoiseRR_stat-trialBetas_desc-zscore_statseries.h5",
    ).resolve()
    betas_h5 = h5py.File(betas_path, "r")

    labels_path = Path(
        f"{idir}/THINGS/behaviour/sub-{sub}/beh/"
        f"sub-{sub}_task-things_desc-perTrial_annotation.tsv",
    ).resolve()
    labels = pd.read_csv(labels_path, sep = '\t', low_memory=False)

    # exclude session 1
    labels = labels[labels["session_id"] != "ses-001"]

    return betas_h5, labels


def build_xy(betas_h5, labels):

    new_labels = pd.DataFrame(columns=labels.columns)
    concat_betas = None

    ses_list = np.unique(labels["session_id"])
    for ses in tqdm.tqdm(ses_list, desc = 'buiding data array'):
        df_ses = labels[labels["session_id"]==ses]
        ses_key = str(int(ses.split("-")[-1]))

        run_list = np.unique(df_ses["run_id"])
        for run in run_list:
            run_df = df_ses[df_ses["run_id"]==run]
            run_key = str(run)

            if run_df.shape[0]==60:
                try:
                    run_betas = np.array(betas_h5[ses_key][run_key]['betas'])
                    if run_betas.shape[0] == 60:
                        concat_betas = run_betas if concat_betas is None else np.concatenate(
                            (concat_betas, run_betas), axis=0,
                        )
                        new_labels = pd.concat(
                            [new_labels, run_df],
                            ignore_index=True,
                            sort=False,
                        )
                except:
                    print(f"no betas for {ses}, run 0{str(run)}")

    # remove not-for-memory sessions
    for_memo = new_labels["not_for_memory"] == False
    concat_betas = concat_betas[for_memo]
    new_labels = new_labels[for_memo]

    # remove atypical==True trials
    typical = new_labels["atypical"] == False
    concat_betas = concat_betas[typical]
    new_labels = new_labels[typical]

    return concat_betas, new_labels


def do_ttest(betas, labels, label_col, lablist_1, lablist_2):

    grp1_mask = labels[label_col].isin(lablist_1)
    betas_grp1 = betas[grp1_mask]

    grp2_mask = labels[label_col].isin(lablist_2)
    betas_grp2 = betas[grp2_mask]

    ttest_res = stats.ttest_ind(
        a=betas_grp1, b=betas_grp2, axis=0, equal_var=True
    )

    return ttest_res


def export_to_brain(idir, sub, tscores, contrast_label, brain_mask):

    nib.save(
        unmask(tscores.statistic, brain_mask),
        f"{idir}/THINGS/glm-memory/sub-{sub}/glm/"
        f"sub-{sub}_task-things_space-T1w_contrast-{contrast_label}_"
        "stat-t_desc-fromBetas_statmap.nii.gz",
    )
    nib.save(
        unmask(tscores.pvalue, brain_mask),
        f"{idir}/THINGS/glm-memory/sub-{sub}/glm/"
        f"sub-{sub}_task-things_space-T1w_contrast-{contrast_label}_"
        "stat-p_desc-fromBetas_statmap.nii.gz",
    )


def perform_ttest(idir, sub):

    betas_h5, labels = load_files(idir, sub)

    betas, labels = build_xy(betas_h5, labels)

    brain_mask = nib.nifti1.Nifti1Image(
        np.array(betas_h5['mask_array']), affine=np.array(betas_h5['mask_affine']),
    )
    betas_h5.close()

    # Hits (all types) versus Correct Rejections
    hit_cr_scores = do_ttest(
        betas, labels, "response_type", ["Hit"], ["CR"],
    )
    export_to_brain(idir, sub, hit_cr_scores, "HitvCorrectRej", brain_mask)

    # Hits within versus Correct Rejections
    hitw_cr_scores = do_ttest(
        betas, labels, "response_subtype", ["Hit_w", "Hit_bw"], ["CR"],
    )
    export_to_brain(idir, sub, hitw_cr_scores, "HitWithinvCorrectRej", brain_mask)

    # Hits between versus Correct Rejections
    hitb_cr_scores = do_ttest(
        betas, labels, "response_subtype", ["Hit_b", "Hit_wb"], ["CR"],
    )
    export_to_brain(idir, sub, hitb_cr_scores, "HitBtwnvCorrectRej", brain_mask)


def main():
    """
    Script performs series of t-tests contrasting trialwise beta scores
    (estimated with GLMSingle) across memory conditions from the THINGS dataset.
    Contrasts are :
    - Hits versus Correct Rejections
    - Hits Within versus Correct Rejections
    - Hits Between versus Correct Rejections
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--idir',
        type=str,
        required=True,
        help='path to cneuromod-things dataset',
    )
    parser.add_argument(
        '--sub',
        required=True,
        type=str,
        help='two-digit subject number',
    )
    args = parser.parse_args()

    perform_ttest(args.idir, args.sub)


if __name__ == '__main__':
    sys.exit(main())
