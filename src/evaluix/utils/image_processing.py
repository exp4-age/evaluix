'''
This module provides image-sequence alignment for Kerr microscopy videos.
It is written blockwise for maintenance:
    1. Import all required external modules
    2. Cost function and single-frame shift estimation
    3. Fourier low-pass filtering (noise suppression before alignment)
    4. Video-level alignment
    5. Example usage
'''
###############################################################################
# 1. Import necessary modules
###############################################################################

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from numpy import fft
from scipy import ndimage
from scipy.optimize import minimize
from skimage import filters
from tqdm import tqdm

# reuse the same logger as the rest of the package
from .file_loader import log_message

#%%
###############################################################################
# 2. Cost function and single-frame shift estimation
###############################################################################

def _alignment_cost(im_ref: np.ndarray, trafo_img: np.ndarray, margin: int = 40):
    """
    Mean squared error between a reference and a shifted image, used as the
    cost function that `estimate_shift` minimizes.

    The outer `margin` pixels are cropped from both images before comparison,
    since `scipy.ndimage.shift` introduces black borders that would otherwise
    bias the optimizer towards under-shifting.

    Parameters
    ----------
    im_ref : numpy.ndarray
        2D reference image.
    trafo_img : numpy.ndarray
        2D shifted image to compare against the reference.
    margin : int, optional
        Number of border pixels excluded from the comparison on each side.
        The default is 40.

    Returns
    -------
    float
        Mean squared error between the cropped images.
    """
    if margin:
        im_ref = im_ref[margin:-margin, margin:-margin]
        trafo_img = trafo_img[margin:-margin, margin:-margin]
    return sklearn.metrics.mean_squared_error(im_ref, trafo_img)

def shift_image(image: np.ndarray, shift_y: float, shift_x: float, axis: str = "both"):
    """
    Shift a 2D image along y and/or x using spline interpolation.

    Parameters
    ----------
    image : numpy.ndarray
        2D image to be shifted.
    shift_y : float
        Shift along the row axis, in pixels.
    shift_x : float
        Shift along the column axis, in pixels.
    axis : {'tb', 'lr', 'both'}, optional
        Which direction(s) to actually shift: 'tb' (top-bottom, y only),
        'lr' (left-right, x only), or 'both'. The default is 'both'.

    Returns
    -------
    numpy.ndarray
        Shifted image, same shape as `image`.

    Raises
    ------
    ValueError
        If `axis` is not one of 'tb', 'lr', 'both'.
    """
    if axis == "tb":
        shift_x = 0
    elif axis == "lr":
        shift_y = 0
    elif axis != "both":
        raise ValueError(f"axis must be 'tb', 'lr', or 'both', not {axis!r}")

    return ndimage.shift(image, (shift_y, shift_x))

def _shift_cost(shift, im_ref, image, axis, margin):
    shift_y, shift_x = shift
    return _alignment_cost(im_ref, shift_image(image, shift_y, shift_x, axis), margin)

def estimate_shift(
    im_ref: np.ndarray,
    image: np.ndarray,
    axis: str = "both",
    initial_guess: tuple = (0, 0),
    method: str = "cobyla",
    margin: int = 40,
):
    """
    Estimate the (y, x) pixel shift that best aligns `image` to `im_ref`, by
    minimizing the mean squared error between them.

    Note: this only returns the estimated shift; it does not shift the image
    itself. Apply it with `shift_image(image, *shift, axis)` if needed.

    Parameters
    ----------
    im_ref : numpy.ndarray
        2D reference image.
    image : numpy.ndarray
        2D image to align to the reference.
    axis : {'tb', 'lr', 'both'}, optional
        Which direction(s) to allow shifting in. The default is 'both'.
    initial_guess : tuple of float, optional
        Starting (shift_y, shift_x) for the optimizer. The default is (0, 0).
    method : str, optional
        Optimization method passed to `scipy.optimize.minimize`.
        The default is 'cobyla'.
    margin : int, optional
        Border pixels excluded from the cost calculation, see
        `_alignment_cost`. The default is 40.

    Returns
    -------
    shift : numpy.ndarray
        The estimated (shift_y, shift_x), in pixels.

    Examples
    --------
    >>> shift = estimate_shift(frame_0, frame_1)  # doctest: +SKIP
    >>> aligned_frame_1 = shift_image(frame_1, *shift)  # doctest: +SKIP
    """
    result = minimize(_shift_cost, initial_guess, args=(im_ref, image, axis, margin), method=method)
    return result["x"]

#%%
###############################################################################
# 3. Fourier low-pass filtering (noise suppression before alignment)
###############################################################################

def _lowpass_mask(shape: tuple, dx: int, dy: int):
    """
    Build a rectangular low-pass mask (ones inside a centered dx-by-dy box,
    zeros elsewhere), for use on an fftshift-ed frequency spectrum of the
    given shape.

    Parameters
    ----------
    shape : tuple of int
        (rows, columns) of the mask, matching the FFT'd image.
    dx : int
        Half-width of the passband along the column axis, in pixels.
    dy : int
        Half-width of the passband along the row axis, in pixels.

    Returns
    -------
    mask : numpy.ndarray
        Array of the given shape, 1 inside the centered box and 0 outside.
    """
    rows, cols = shape
    y_mid, x_mid = rows // 2, cols // 2

    y_max = y_mid + dy - (0 if rows % 2 else 1)
    x_max = x_mid + dx - (0 if cols % 2 else 1)

    mask = np.zeros(shape)
    mask[max(y_mid - dy, 0):y_max + 1, max(x_mid - dx, 0):x_max + 1] = 1
    return mask

def enhance(img: np.ndarray, dx: int = 100, dy: int = 100):
    """
    Apply a rectangular low-pass filter in Fourier space to suppress
    high-frequency noise, as a preprocessing step before shift estimation.

    Parameters
    ----------
    img : numpy.ndarray
        2D grayscale image.
    dx : int, optional
        Half-width of the passband along x, in pixels. The default is 100.
    dy : int, optional
        Half-width of the passband along y, in pixels. The default is 100.

    Returns
    -------
    numpy.ndarray
        Filtered image, same shape as `img`, real-valued.
    """
    spectrum = fft.fftshift(fft.fft2(img))
    spectrum = spectrum * _lowpass_mask(spectrum.shape, dx, dy)
    return np.abs(fft.ifft2(spectrum))

#%%
###############################################################################
# 4. Video-level alignment
###############################################################################

def _plot_shift_trajectory(cumulative_shift: np.ndarray, save_path: str | None = None):
    frames = np.arange(len(cumulative_shift))
    fig, ax = plt.subplots(1, 2, figsize=(16 / 2.54, 8 / 2.54))
    ax[0].plot(frames, cumulative_shift[:, 0], c='tab:blue', label='y shift')
    ax[0].plot(frames, cumulative_shift[:, 1], c='tab:orange', label='x shift')
    ax[0].set_xlabel('frame')
    ax[0].set_ylabel('shift [px]')
    ax[0].legend()
    ax[1].plot(cumulative_shift[:, 1], cumulative_shift[:, 0], c='black')
    ax[1].set_xlabel('x shift [px]')
    ax[1].set_ylabel('y shift [px]')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def align_video(
    video: list,
    axis: str = "both",
    crop_margin: int = 5,
    plot: bool = True,
    save_path: str | None = "shift.png",
):
    """
    Align every frame of a video sequence to the first frame by iteratively
    estimating and applying frame-to-frame shifts.

    Each frame is aligned to its (low-pass filtered) predecessor, and the
    resulting shifts are accumulated so that every frame ends up aligned to
    frame 0, correcting for sample drift, focus-motor jitter, or a moving
    field of view over the course of a measurement.

    Parameters
    ----------
    video : list of numpy.ndarray
        Sequence of 2D frames to align, e.g. as returned by
        `file_loader.read_image_sequence` or `file_loader.read_video`.
    axis : {'tb', 'lr', 'both'}, optional
        Which direction(s) to allow shifting in. The default is 'both'.
    crop_margin : int, optional
        Pixels cropped from each border of every aligned frame, to remove the
        black border introduced by shifting. The default is 5.
    plot : bool, optional
        If True (default), plot the shift-per-frame and the x-y shift
        trajectory, useful for spotting drift or a bad alignment.
    save_path : str, optional
        Where to save the diagnostic plot, only used if `plot` is True.
        The default is 'shift.png'. Set to None to display the plot instead
        of saving it.

    Returns
    -------
    aligned_video : list of numpy.ndarray
        Cropped, aligned frames, same length as `video`.
    cumulative_shift : numpy.ndarray
        Array of shape (len(video), 2) with the total (shift_y, shift_x)
        applied to each frame relative to frame 0 - i.e. the shift that was
        actually used to produce `aligned_video`, not the raw frame-to-frame
        shift.

    Examples
    --------
    >>> aligned_video, shift = align_video(video)  # doctest: +SKIP
    """
    log_message('info', f"Aligning a video of {len(video)} frames along axis={axis!r}.")

    aligned_video = [video[0][crop_margin:-crop_margin, crop_margin:-crop_margin]]
    cumulative_shift = np.zeros((len(video), 2))

    for i in tqdm(range(1, len(video)), desc="Aligning frames"):
        frame_shift = estimate_shift(
            filters.gaussian(enhance(video[i - 1]), 2),
            filters.gaussian(enhance(video[i]), 2),
            axis=axis,
        )
        cumulative_shift[i] = cumulative_shift[i - 1] + frame_shift
        aligned_frame = ndimage.shift(video[i], cumulative_shift[i])
        aligned_video.append(aligned_frame[crop_margin:-crop_margin, crop_margin:-crop_margin])

    if plot:
        _plot_shift_trajectory(cumulative_shift, save_path)

    log_message('info', "Video alignment complete.")
    return aligned_video, cumulative_shift

#%%
###############################################################################
# 5. Example usage
###############################################################################
# Kept here (inactive) as a worked example - edit the path and run this file
# directly (or copy the lines into your own script/notebook) to use it.

if __name__ == "__main__":
    from .file_loader import read_image_sequence

    folder = r"path/to/your/frame_folder"  # <- replace with your data folder
    video = read_image_sequence(folder, gray=False)

    aligned_video, shift = align_video(video[:-1], axis="both")

    # save the aligned video for later use, so the alignment does not need to be repeated
    np.save("shifted_video.npy", aligned_video)
