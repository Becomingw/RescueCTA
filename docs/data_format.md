# Data format

## Training pairs

`SliceDataset` reads paired `.npy` or `.dcm` files from the directories named by
`source_dir` and `cta_dir` in the YAML configuration. Directory entries must
match exactly so that each source image has one spatially registered CTA
target.

Accepted array layouts are:

- `H × W` for a single section;
- `H × W × 3` for a 2.5D stack; or
- `3 × H × W`, which is converted internally.

The manuscript model uses three adjacent sections as three channels. Images
must be expressed in Hounsfield units. Preprocessing applies a joint foreground
bounding box, bilinear resizing to `512 × 512`, and

```text
x_norm = 2 × max(x_HU + 1024, 0) / 4095 - 1.
```

The original study paired the source and target images through deformable
registration before creating these training arrays. Registration is outside
the scope of this repository.

## Inference volumes

`infer_nifti.py` accepts `.nii` and `.nii.gz` volumes with shape `H × W × D`
and intensity values in Hounsfield units. It preserves the affine and header,
uses adjacent sections to form each 2.5D input, restores the original in-plane
matrix, and writes compressed NIfTI output. The default channel-reduction
strategy is the historical three-channel mean; central-channel output is
available through `--channel_strategy center`.
