""" Python script to evaluate super resolved images against ground truth high resolution images """

import itertools
import cv2

import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

from DataLoader import get_patch


def cMSE(sr, hr, hr_map):

    n_clear = np.sum(hr_map, axis=(1, 2))  # number of clear pixels in the high-res patch
    diff = hr - sr
    bias = np.sum(diff * hr, axis=(1, 2)) / n_clear  # brightness bias
    cMSE = np.sum(np.square((diff - bias[:, None, None]) * hr_map), axis=(1, 2)) / n_clear

    return cMSE

def shift_cMSE(sr, hr, hr_map, border_w=3):
    """
    cPSNR score adjusted for registration errors. Computes the max cPSNR score across shifts of up to `border_w` pixels.
    Args:
        sr: np.ndarray (n, m), super-resolved image
        hr: np.ndarray (n, m), high-res ground-truth image
        hr_map: np.ndarray (n, m), high-res status map
        border_w: int, width of the trimming border around `hr` and `hr_map`
    Returns:
        max_cPSNR: float, score of the super-resolved image
    """

    size = sr.shape[1] - (2 * border_w)  # patch size
    sr = get_patch(img=sr, x=border_w, y=border_w, size=size)
    pos = list(itertools.product(range(2 * border_w + 1), range(2 * border_w + 1)))
    iter_hr = patch_iterator(img=hr, positions=pos, size=size)
    iter_hr_map = patch_iterator(img=hr_map, positions=pos, size=size)
    site_cMSE = np.array([cMSE(sr, hr, hr_map) for hr, hr_map in tqdm(zip(iter_hr, iter_hr_map),
                                                                        disable=(len(sr.shape) == 2))
                           ])
    min_cMSE = np.min(site_cMSE, axis=0)
    return min_cMSE

def cPSNR(sr, hr, hr_map):
    """
    Clear Peak Signal-to-Noise Ratio. The PSNR score, adjusted for brightness and other volatile features, e.g. clouds.
    Args:
        sr: numpy.ndarray (n, m), super-resolved image
        hr: numpy.ndarray (n, m), high-res ground-truth image
        hr_map: numpy.ndarray (n, m), status map of high-res image, indicating clear pixels by a value of 1
    Returns:
        cPSNR: float, score
    """

 #   if len(sr.shape) == 2:
 #       sr = sr[None, ]
 #       hr = hr[None, ]
 #       hr_map = hr_map[None, ]

 #   if sr.dtype.type is np.uint16:  # integer array is in the range [0, 65536]
 #       sr = sr / np.iinfo(np.uint16).max  # normalize in the range [0, 1]
 #   else:
 #       assert 0 <= sr.min() and sr.max() <= 1, 'sr.dtype must be either uint16 (range 0-65536) or float64 in (0, 1).'
  #  if hr.dtype.type is np.uint16:
   #     hr = hr / np.iinfo(np.uint16).max

    n_clear = np.sum(hr_map, axis=(1, 2))  # number of clear pixels in the high-res patch
    diff = hr - sr
    bias = np.sum(diff * hr_map, axis=(1, 2)) / n_clear  # brightness bias
    val_cMSE = np.sum(np.square((diff - bias[:, None, None]) * hr_map), axis=(1, 2)) / n_clear
    val_cPSNR = -10 * np.log10(val_cMSE)  # + 1e-10)

    if val_cPSNR.shape[0] == 1:
        val_cPSNR = val_cPSNR[0]

    return val_cPSNR



def patch_iterator(img, positions, size):
    """Iterator across square patches of `img` located in `positions`."""
    for x, y in positions:
        yield get_patch(img=img, x=x, y=y, size=size)


def shift_cPSNR(sr, hr, hr_map, border_w=3):
    """
    cPSNR score adjusted for registration errors. Computes the max cPSNR score across shifts of up to `border_w` pixels.
    Args:
        sr: np.ndarray (n, m), super-resolved image
        hr: np.ndarray (n, m), high-res ground-truth image
        hr_map: np.ndarray (n, m), high-res status map
        border_w: int, width of the trimming border around `hr` and `hr_map`
    Returns:
        max_cPSNR: float, score of the super-resolved image
    """

    size = sr.shape[1] - (2 * border_w)  # patch size
    sr = get_patch(img=sr, x=border_w, y=border_w, size=size)
    pos = list(itertools.product(range(2 * border_w + 1), range(2 * border_w + 1)))
    iter_hr = patch_iterator(img=hr, positions=pos, size=size)
    iter_hr_map = patch_iterator(img=hr_map, positions=pos, size=size)
    site_cPSNR = np.array([cPSNR(sr, hr, hr_map) for hr, hr_map in tqdm(zip(iter_hr, iter_hr_map),
                                                                        disable=(len(sr.shape) == 2))
                           ])
    max_cPSNR = np.max(site_cPSNR, axis=0)
    return max_cPSNR

def cSSIM(sr, hr):

    #if len(sr.shape) == 2:
    #    sr = sr[None, ]
    #    hr = hr[None, ]

   # if sr.dtype.type is np.uint16:  # integer array is in the range [0, 65536]
   #     sr = sr / np.iinfo(np.uint16).max  # normalize in the range [0, 1]
   # else:
   #     assert 0 <= sr.min() and sr.max() <= 1, 'sr.dtype must be either uint16 (range 0-65536) or float64 in (0, 1).'
   # if hr.dtype.type is np.uint16:
   #     hr = hr / np.iinfo(np.uint16).max

    cSSIM = ssim(sr[None, :, :], hr[None, :, :], multichannel=True, data_range=1.0)

    return cSSIM

def shift_SSIM(sr, hr, hr_map):
    pass