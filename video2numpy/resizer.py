"""
  Initial version, just to reproduce previous state

  TODO: implement any shape1 -> any shape2 vs. just any shape1 -> (size, size)
"""

import cv2


class Resizer:
    """
    Class for resizing frames to uniform shape
    """

    def __init__(self, from_shape, to_size):
        self.from_shape = from_shape
        self.to_size = to_size

        sm_ind, bg_ind = (0, 1) if from_shape[0] < from_shape[1] else (1, 0)
        ratio = from_shape[sm_ind] / to_size
        n_shape = from_shape
        n_shape[sm_ind] = to_size
        n_shape[bg_ind] = max(int(n_shape[bg_ind] / ratio), to_size)  # safety for rounding errors

        self.resize_shape = tuple(n_shape)

    def __call__(self, img):
        # Resize:
        resized = cv2.resize(img, (self.resize_shape[1], self.resize_shape[0]), interpolation=cv2.INTER_CUBIC)

        # Center crop:
        my = int((self.resize_shape[0] - self.to_size) / 2)
        mx = int((self.resize_shape[1] - self.to_size) / 2)

        cropped = resized[my : resized.shape[0] - my, mx : resized.shape[1] - mx]
        cropped = cropped[: self.to_size, : self.to_size]  # safety from approx.

        return cropped
