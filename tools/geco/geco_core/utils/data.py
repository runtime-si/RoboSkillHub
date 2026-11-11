# ruff: noqa  (vendored from GeCo @ c0fad20, MIT)
# From GeCo (https://github.com/jerpelhan/GeCo @ c0fad20), utils/data.py, MIT, Copyright (c) 2024 jerpelhan. Only resize_and_pad kept.
import torch


def resize_and_pad(
    img,
    bboxes,
    density_map=None,
    size=1024.0,
    gt_bboxes=None,
    full_stretch=True,
    downscale_factor=1,
):
    channels, original_height, original_width = img.shape
    longer_dimension = max(original_height, original_width)
    scaling_factor = size / longer_dimension

    if not full_stretch:
        scaled_bboxes = bboxes * scaling_factor

        a_dim = (
            (scaled_bboxes[:, 2] - scaled_bboxes[:, 0]).mean()
            + (scaled_bboxes[:, 3] - scaled_bboxes[:, 1]).mean()
        ) / 2
        scaling_factor = min(1.0, 80 / a_dim.item()) * scaling_factor

    if downscale_factor != 1:
        scaling_factor = scaling_factor * downscale_factor

    resized_img = torch.nn.functional.interpolate(
        img.unsqueeze(0), scale_factor=scaling_factor, mode="bilinear", align_corners=False
    )

    if max(resized_img.shape) <= 1024:
        size = 1024
    size = int(size)
    pad_height = max(0, size - resized_img.shape[2])
    pad_width = max(0, size - resized_img.shape[3])

    padded_img = torch.nn.functional.pad(
        resized_img, (0, pad_width, 0, pad_height), mode="constant", value=0
    )[0]

    if density_map is not None:
        original_sum = density_map.sum()
        _, w0, h0 = density_map.shape
        _, W, H = img.shape
        resized_density_map = torch.nn.functional.interpolate(
            density_map.unsqueeze(0), size=(W, H), mode="bilinear", align_corners=False
        )
        resized_density_map = torch.nn.functional.interpolate(
            resized_density_map, scale_factor=scaling_factor, mode="bilinear", align_corners=False
        )
        padded_density_map = torch.nn.functional.pad(
            resized_density_map, (0, pad_width, 0, pad_height), mode="constant", value=0
        )[0]
        padded_density_map = T.Resize((512, 512), antialias=True)(padded_density_map)
        padded_density_map = padded_density_map / padded_density_map.sum() * original_sum

    bboxes = bboxes * torch.tensor([scaling_factor, scaling_factor, scaling_factor, scaling_factor])

    if gt_bboxes is not None and density_map is not None:
        gt_bboxes = gt_bboxes * torch.tensor(
            [scaling_factor, scaling_factor, scaling_factor, scaling_factor]
        )
        return (
            padded_img,
            bboxes,
            padded_density_map,
            gt_bboxes,
            scaling_factor,
            (pad_width, pad_height),
        )
    if gt_bboxes is not None:
        return padded_img, bboxes, gt_bboxes, scaling_factor, (pad_width, pad_height)
    if density_map is None and gt_bboxes is None:
        return padded_img, bboxes, scaling_factor

    return padded_img, bboxes, padded_density_map
