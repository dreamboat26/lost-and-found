'''
 # @ Author: Meet Patel
 # @ Create Time: 2024-11-07 05:30:00
 # @ Modified by: Meet Patel
 # @ Modified time: 2024-11-09 09:06:22
 # @ Description:
 '''

from torcheval.metrics.functional import peak_signal_noise_ratio
from torchmetrics.image import StructuralSimilarityIndexMeasure

def get_psnr(img1, img2, data_range) -> float:
    psnr_value = peak_signal_noise_ratio(img1, img2, data_range)
    return psnr_value

def get_ssim(img1, img2, data_range) -> float:
    ssim_value = StructuralSimilarityIndexMeasure(data_range=data_range)(img1, img2)
    return ssim_value