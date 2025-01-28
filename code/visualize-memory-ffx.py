import os
from pathlib import Path

import h5py
import click
import cortex
import numpy as np
import nibabel as nib
from nilearn import image, plotting
import matplotlib.pyplot as plt

# required for pycortex on MacOSX (emdupre hardware)
# note that this will silently fail if run on anyone else's env
os.environ["PATH"] += ":/Applications/Inkscape.app/Contents/MacOS/"


@click.command()
@click.option("--sub_name", default="sub-01", help="Subject name.")
@click.option("--stat_type", default="z", help="Stat type for visualization.")
@click.option("--data_dir", default="/Users/emdupre/Desktop", help="Data directory.")
def main(sub_name, stat_type, data_dir):
    """ """
    sub_names = ["sub-01", "sub-02", "sub-03", "sub-06"]
    if sub_name not in sub_names:
        warn_msg = "Unrecognized subject {sub_name}"
        raise UserWarning(warn_msg)

    stat_types = ["effect", "variance", "t", "z"]
    if stat_type not in stat_types:
        warn_msg = "Unrecognized stat type {stat_type}"
        raise UserWarning(warn_msg)

    # mask_fname = Path(data_dir, "glmsingle", sub_name).rglob(
    #     f"{sub_name}_task-things_space-T1w_label-brain_desc-unionNonNaN_mask.nii"
    # )
    # mask = nib.nifti1.Nifti1Image(mask_fname)
    beta_fname = f"{sub_name}_task-things_space-T1w_model-fitHrfGLMdenoiseRR_stats-imageBetas_desc-zscore_statseries.h5"
    beta_h5 = h5py.File(Path(data_dir, "things-encode", "betas", beta_fname), "r")
    mask = nib.nifti1.Nifti1Image(
        np.array(beta_h5["mask_array"]), affine=np.array(beta_h5["mask_affine"])
    )

    ffx_fname = f"{sub_name}_task-things_space-T1w_contrast-HitBtwnvCorrectRej_stat-{stat_type}_statmap.nii.gz"
    anat_fname = f"{sub_name}_desc-preproc_T1w.nii.gz"
    try:
        # ffx_nii = nib.load(
        #     Path(data_dir, "things-glm", "things-glm", sub_name, "glm", ffx_fname)
        # )
        ffx_nii = nib.load(Path(data_dir, "things-glm", "things-glm", ffx_fname))
    except FileNotFoundError:
        warn_msg = (
            f"Statmap not found for subject {sub_name}. "
            f"Please make sure you have previously run `gen_memory_ffx.py` for {sub_name}"
        )
        raise UserWarning(warn_msg)
    try:
        anat_nii = nib.load(
            Path(
                data_dir,
                "things-glm",
                "things.fmriprep",
                "sourcedata",
                "smriprep",
                sub_name,
                "anat",
                anat_fname,
            )
        )
    except FileNotFoundError:
        warn_msg = (
            "Anatomical files cannot be loaded. Please ensure that files have been "
            "first downloaded with datalad."
        )
        raise UserWarning(warn_msg)

    # plotting.plot_stat_map(
    #     ffx_nii,
    #     bg_img=None,
    #     threshold="auto",
    #     display_mode="z",
    #     cut_coords=3,
    #     black_bg=True,
    #     title="hit_within-correct_rej",
    # )

    plotting.view_img(
        ffx_nii,
        bg_img=anat_nii,
        threshold="auto",
        black_bg=True,
    ).save_as_html(
        f"{sub_name}_task-things_space-T1w_contrast-HitBtwnvCorrectRej_stat-{stat_type}_statmap.html"
    )

    ffx_vol = cortex.Volume(
        data=np.swapaxes(ffx_nii.get_fdata(), 0, -1),
        subject=sub_name,
        xfmname="align_auto",
        mask=mask.get_fdata(),
        # vmin=(np.min(ffx_nii.dataobj) * 0.95),
        # vmax=(np.max(ffx_nii.dataobj) * 0.95),
        # cmap="magma",
    )
    cortex.quickflat.make_png(
        f"{sub_name}_task-things_space-T1w_contrast-HitBtwnvCorrectRej_stat-{stat_type}_flatmap.png",
        ffx_vol,
        sampler="trilinear",
        curv_brightness=1.0,
        with_colorbar=True,
        colorbar_location="left",
        with_curvature=True,
        with_labels=False,
        with_rois=True,
        dpi=300,
        height=2048,
    )


if __name__ == "__main__":
    main()
