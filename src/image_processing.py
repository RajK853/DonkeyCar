import cv2
import copy
import numpy as np


def crop_img(img, crop_dim):
    assert all(isinstance(crop, slice) for crop in crop_dim), f"Crop_dim should be an slice, not {type(crop_dim)}"
    img = img[crop_dim]
    return img


def flip_img(img):
    flipped_img = cv2.flip(img, 1)
    return flipped_img


def blur_img(img, ksize=None):
    ksize = ksize or (3, 3)
    blurred_img = cv2.GaussianBlur(img, ksize, 0)
    return blurred_img


def illuminate_img(img, alpha=1, beta=50):
    result_img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
    return result_img


def normalize_images(images):
    return images/255.0


def denormalize_images(images):
    images = images * 255.0
    return images.astype(np.uint8)
