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
    sub_names = ["sub-01", "sub-02", "sub-03"]
    if sub_name not in sub_names:
        warn_msg = "Unrecognized subject {sub_name}"
        raise UserWarning(warn_msg)

    stat_types = ["effect", "variance", "t", "z"]
    if stat_type not in stat_types:
        warn_msg = "Unrecognized stat type {stat_type}"
        raise UserWarning(warn_msg)

    beta_fname = f"{sub_name}_task-things_space-T1w_model-fitHrfGLMdenoiseRR_stats-trialBetas_desc-zscore_statseries.h5"
    beta_h5 = h5py.File(Path(data_dir, "things-encode", "betas", beta_fname), "r")
    mask = nib.nifti1.Nifti1Image(
        np.array(beta_h5["mask_array"]), affine=np.array(beta_h5["mask_affine"])
    )

    ffx_fname = f"{sub_name}_task-things_space-T1w_contrast-HitWithinvCorrectRej_stat-{stat_type}_statmap.nii.gz"
    try:
        ffx_nii = nib.load(Path(data_dir, "things-glm", ffx_fname))
    except FileNotFoundError:
        warn_msg = (
            f"Statmap not found for subject {sub_name}. "
            f"Please make sure you have previously run `gen_memory_ffx.py` for {sub_name}"
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
        bg_img=None,
        threshold="auto",
        black_bg=True,
    ).save_as_html(
        f"{sub_name}_task-things_space-T1w_contrast-HitWithinvCorrectRej_stat-{stat_type}_statmap.html"
    )

    ffx_vol = cortex.Volume(
        data=np.swapaxes(ffx_nii.get_fdata(), 0, -1),
        subject=sub_name,
        xfmname="align_auto",
        mask=mask.get_fdata(),
        vmin=(np.max(ffx_nii.dataobj) * 0.05),
        vmax=(np.max(ffx_nii.dataobj) * 0.95),
        cmap="magma",
    )
    # _ = cortex.quickflat.make_figure(
    #     ffx_vol,
    #     with_colorbar=True,
    #     colorbar_location="left",
    #     with_curvature=True,
    #     sampler="trilinear",
    #     with_labels=False,
    #     with_rois=True,
    #     curv_brightness=1.0,
    #     dpi=300,
    # )
    cortex.quickflat.make_png(
        f"{sub_name}_task-things_space-T1w_contrast-HitWithinvCorrectRej_stat-{stat_type}_flatmap.png",
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
