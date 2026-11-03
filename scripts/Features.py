"""Leaf and tea-bud segmentation and feature extraction."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2 as cv
import numpy as np

MORPHOLOGY_KERNEL = np.ones((5, 5), dtype=np.uint8)


def two_peak_threshold(gray: np.ndarray) -> np.ndarray:
    """Create a binary image using the valley between two histogram peaks."""
    if gray is None or gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("expected an 8-bit grayscale image")
    histogram = np.bincount(gray.ravel(), minlength=256)
    first = int(np.argmax(histogram))
    distances = (np.arange(256) - first) ** 2 * histogram
    second = int(np.argmax(distances))
    low, high = sorted((first, second))
    threshold = low + int(np.argmin(histogram[low:high])) + 1 if high > low else 0
    if threshold > 40:
        threshold = 50
    _, binary = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
    return binary


def preprocess(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return grayscale and morphology-cleaned binary images."""
    if image is None or image.ndim != 3:
        raise ValueError("expected a non-empty BGR image")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (9, 9), 0)
    binary = two_peak_threshold(blurred)
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, MORPHOLOGY_KERNEL)
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, MORPHOLOGY_KERNEL)
    return gray, binary


def largest_contour(contours) -> np.ndarray:
    """Return the largest contour or raise a useful segmentation error."""
    if not contours:
        raise ValueError("no foreground contour found; check image contrast/background")
    return max(contours, key=cv.contourArea)


def shape_features(binary: np.ndarray, dpi: float = 300) -> list[float]:
    """Extract nine geometric features, with physical values based on DPI."""
    if dpi <= 0:
        raise ValueError("dpi must be greater than zero")
    pixels_per_cm = dpi / 2.54
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = largest_contour(contours)
    perimeter = cv.arcLength(contour, True)
    area = cv.contourArea(contour)
    _, _, width, height = cv.boundingRect(contour)
    length_px, width_px = max(width, height), min(width, height)
    rectangle_area = width * height
    hull_area = cv.contourArea(cv.convexHull(contour))
    equivalent_diameter = np.sqrt(4 * area / np.pi)
    circularity = 4 * np.pi * area / perimeter**2 if perimeter else 0.0
    return [
        round(perimeter / pixels_per_cm, 2),
        round(area / pixels_per_cm**2, 2),
        round(length_px / pixels_per_cm, 2),
        round(width_px / pixels_per_cm, 2),
        round(length_px / width_px, 2) if width_px else 0.0,
        round(area / rectangle_area, 2) if rectangle_area else 0.0,
        round(area / hull_area, 2) if hull_area else 0.0,
        round(equivalent_diameter / pixels_per_cm, 2),
        round(circularity, 2),
    ]


def foreground_mask(image: np.ndarray, binary: np.ndarray | None = None) -> np.ndarray:
    """Return a filled mask for the largest foreground object."""
    if image is None or image.ndim != 3:
        raise ValueError("expected a non-empty BGR image")
    if binary is None:
        _, binary = preprocess(image)
    elif binary.shape != image.shape[:2]:
        raise ValueError("foreground binary mask and image dimensions differ")
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv.drawContours(mask, [largest_contour(contours)], -1, 255, cv.FILLED)
    if not np.any(mask):
        raise ValueError("detected foreground region contains no pixels")
    return mask


def interior_mask(mask: np.ndarray, erosion: int = 3) -> np.ndarray:
    """Reduce segmentation-edge contamination without losing tiny objects."""
    if erosion <= 0:
        return mask
    interior = cv.erode(mask, np.ones((3, 3), np.uint8), iterations=erosion)
    return interior if np.any(interior) else mask


def texture_features(image: np.ndarray, levels: int = 32,
                     mask: np.ndarray | None = None, erosion: int = 3) -> list[float]:
    """Extract seven multi-direction GLCM features inside the foreground."""
    if levels not in (8, 16, 32, 64, 128, 256):
        raise ValueError("texture levels must be one of 8,16,32,64,128,256")
    if image is None:
        raise ValueError("expected a non-empty image")
    if image.ndim == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        mask = foreground_mask(image) if mask is None else mask
    elif image.ndim == 2:
        gray = image
        mask = np.full(gray.shape, 255, np.uint8) if mask is None else mask
    else:
        raise ValueError("texture input must be a gray or BGR image")
    if mask.shape != gray.shape:
        raise ValueError("texture mask and image dimensions differ")
    mask = interior_mask(mask, erosion)
    quantized = np.minimum(gray.astype(np.uint16) * levels // 256, levels - 1)
    matrix = np.zeros((levels, levels), dtype=np.float64)
    for dy, dx in ((0, 1), (1, 1), (1, 0), (1, -1)):
        y1 = slice(max(0, -dy), gray.shape[0] - max(0, dy))
        y2 = slice(max(0, dy), gray.shape[0] - max(0, -dy))
        x1 = slice(max(0, -dx), gray.shape[1] - max(0, dx))
        x2 = slice(max(0, dx), gray.shape[1] - max(0, -dx))
        valid = (mask[y1, x1] != 0) & (mask[y2, x2] != 0)
        first, second = quantized[y1, x1][valid], quantized[y2, x2][valid]
        if first.size:
            np.add.at(matrix, (first, second), 1)
            np.add.at(matrix, (second, first), 1)
    if not matrix.sum():
        raise ValueError("not enough foreground pixels for texture extraction")
    probability = matrix / matrix.sum()
    i, j = np.indices(probability.shape)
    difference = i - j
    mean_i, mean_j = np.sum(i * probability), np.sum(j * probability)
    std_i = np.sqrt(np.sum((i - mean_i)**2 * probability))
    std_j = np.sqrt(np.sum((j - mean_j)**2 * probability))
    asm = np.sum(probability**2)
    nonzero = probability[probability > 0]
    values = (
        np.sum(probability * difference**2),
        np.sum(probability * np.abs(difference)),
        np.sum(probability / (1.0 + difference**2)),
        (np.sum((i - mean_i) * (j - mean_j) * probability) / (std_i * std_j)
         if std_i and std_j else 1.0),
        asm,
        np.sqrt(asm),
        -np.sum(nonzero * np.log2(nonzero)),
    )
    return [round(float(value), 6) for value in values]


def foreground_pixels(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Return an N×3 array of BGR pixels inside the foreground."""
    mask = foreground_mask(image) if mask is None else mask
    pixels = image[mask != 0]
    if not pixels.size:
        raise ValueError("detected foreground region contains no pixels")
    return pixels


def channel_moments(values: np.ndarray) -> list[float]:
    """Return mean, population standard deviation and cube-root third moment."""
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    return [round(mean, 4), round(float(np.std(values)), 4),
            round(float(np.cbrt(np.mean((values - mean)**3))), 4)]


def color_features(image: np.ndarray, mask: np.ndarray | None = None,
                   erosion: int = 3) -> list[float]:
    """Extract color moments in HSV, RGB and CIE Lab foreground pixels."""
    mask = foreground_mask(image) if mask is None else mask
    pixels = foreground_pixels(image, interior_mask(mask, erosion)).reshape(1, -1, 3)
    hsv = cv.cvtColor(pixels, cv.COLOR_BGR2HSV).astype(np.float64)[0]
    hsv[:, 0] *= 2.0
    hsv[:, 1:] *= 100.0 / 255.0
    rgb = cv.cvtColor(pixels, cv.COLOR_BGR2RGB).astype(np.float64)[0]
    lab = cv.cvtColor(pixels, cv.COLOR_BGR2LAB).astype(np.float64)[0]
    lab[:, 0] *= 100.0 / 255.0
    lab[:, 1:] -= 128.0
    result = []
    for color_space in (hsv, rgb, lab):
        for channel in range(3):
            result.extend(channel_moments(color_space[:, channel]))
    return result


def trichome_index(image: np.ndarray, hmin: int, smin: int, vmin: int,
                   hmax: int, smax: int, vmax: int) -> list[float]:
    """Return matching pixel count, foreground count and percentage."""
    pixels = foreground_pixels(image)
    hsv = cv.cvtColor(pixels.reshape(1, -1, 3), cv.COLOR_BGR2HSV)
    selected = cv.inRange(hsv, np.array([hmin, smin, vmin], np.uint8),
                         np.array([hmax, smax, vmax], np.uint8))
    count = int(np.count_nonzero(selected))
    total = int(len(pixels))
    return [count, total, count / total * 100.0]


def hsv_pixels(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return foreground H, S and V channel arrays."""
    pixels = foreground_pixels(image)
    hsv = cv.cvtColor(pixels.reshape(1, -1, 3), cv.COLOR_BGR2HSV)[0]
    return hsv[:, 0], hsv[:, 1], hsv[:, 2]


def trichome_mask(image: np.ndarray, hmin: int, smin: int, vmin: int,
                  hmax: int, smax: int, vmax: int) -> np.ndarray:
    """Keep image pixels inside an HSV interval."""
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, np.array([hmin, smin, vmin]),
                      np.array([hmax, smax, vmax]))
    return cv.bitwise_and(image, image, mask=mask)


def write_csv(path: Path, rows, headers) -> None:
    """Write a UTF-8-BOM CSV that opens cleanly in spreadsheet software."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
