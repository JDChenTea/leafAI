"""Reusable object detection, annotation and image splitting operations."""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class DetectedObject:
    contour: np.ndarray
    area: float
    bounds: tuple[int, int, int, int]
    rotation: float


def major_axis_rotation(contour: np.ndarray) -> float:
    """Return the small rotation that makes a contour's major axis vertical."""
    points = contour[:, 0, :].astype(np.float64)
    if len(points) < 2:
        return 0.0
    covariance = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    major_axis = vectors[:, int(np.argmax(values))]
    angle = np.degrees(np.arctan2(major_axis[1], major_axis[0])) % 180.0
    return float(angle - 90.0)


def two_peak_threshold(gray: np.ndarray, fallback: int = 25,
                       valid_range: tuple[int, int] = (15, 40)) -> tuple[int, np.ndarray]:
    """Threshold a gray image using the valley between two histogram peaks."""
    if gray is None or gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("two_peak_threshold expects an 8-bit grayscale image")
    histogram = np.bincount(gray.ravel(), minlength=256)
    first = int(np.argmax(histogram))
    distances = (np.arange(256) - first) ** 2 * histogram
    second = int(np.argmax(distances))
    low, high = sorted((first, second))
    threshold = low + int(np.argmin(histogram[low:high])) + 1 if high > low else fallback
    if not valid_range[0] <= threshold <= valid_range[1]:
        threshold = fallback
    _, binary = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
    return threshold, binary


def detect_objects(image: np.ndarray, min_area: float, max_area: float,
                   border: int = 0) -> tuple[np.ndarray, int, list[DetectedObject]]:
    """Detect foreground objects and return the padded image, threshold and objects."""
    if image is None or image.ndim != 3:
        raise ValueError("expected a non-empty BGR image")
    if min_area < 0 or max_area <= min_area:
        raise ValueError("area limits must satisfy 0 <= min-area < max-area")
    work = cv.copyMakeBorder(image, border, border, border, border,
                             cv.BORDER_CONSTANT, value=(0, 0, 0)) if border else image.copy()
    gray = cv.cvtColor(work, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (9, 9), 0)
    threshold, binary = two_peak_threshold(blurred)
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    cleaned = cv.morphologyEx(cleaned, cv.MORPH_OPEN, kernel)
    contours, _ = cv.findContours(cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    objects = []
    for contour in contours:
        area = float(cv.contourArea(contour))
        if not min_area < area < max_area:
            continue
        x, y, width, height = cv.boundingRect(contour)
        rotation = major_axis_rotation(contour)
        objects.append(DetectedObject(contour, area, (x, y, width, height), rotation))

    # Stable output numbering: top-to-bottom, then left-to-right.
    objects.sort(key=lambda item: (item.bounds[1], item.bounds[0]))
    return work, threshold, objects


def annotate_objects(image: np.ndarray, min_area: float = 3000,
                     max_area: float = 700000) -> tuple[np.ndarray, int, int]:
    work, threshold, objects = detect_objects(image, min_area, max_area)
    annotated = work.copy()
    scale = max(0.6, min(2.0, max(image.shape[:2]) / 1200.0))
    thickness = max(2, round(scale * 2))
    for number, item in enumerate(objects, 1):
        x, y, width, height = item.bounds
        cv.rectangle(annotated, (x, y), (x + width, y + height),
                     (255, 255, 255), thickness)
        cv.putText(annotated, str(number), (x, max(20, y - 10)),
                   cv.FONT_HERSHEY_COMPLEX, scale, (255, 255, 255),
                   thickness, cv.LINE_AA)
    return annotated, threshold, len(objects)


def split_objects(image: np.ndarray, min_area: float = 3000,
                  max_area: float = 700000, border: int = 500,
                  margin: int = 50) -> tuple[list[np.ndarray], int]:
    """Detect, deskew and crop every foreground object in one image."""
    if border < 0 or margin < 0:
        raise ValueError("border and margin must be non-negative")
    work, threshold, objects = detect_objects(image, min_area, max_area, border)
    crops = []
    for item in objects:
        x, y, width, height = item.bounds
        center = (x + width / 2.0, y + height / 2.0)
        rotation_matrix = cv.getRotationMatrix2D(center, item.rotation, 1.0)
        rotated = cv.warpAffine(work, rotation_matrix, (work.shape[1], work.shape[0]),
                                flags=cv.INTER_LINEAR, borderValue=(0, 0, 0))
        rotated_contour = cv.transform(item.contour.astype(np.float32), rotation_matrix)
        rotated_x, rotated_y, rotated_width, rotated_height = cv.boundingRect(
            rotated_contour.astype(np.int32)
        )
        x1, y1 = max(0, rotated_x - margin), max(0, rotated_y - margin)
        x2 = min(rotated.shape[1], rotated_x + rotated_width + margin)
        y2 = min(rotated.shape[0], rotated_y + rotated_height + margin)
        crop = rotated[y1:y2, x1:x2]
        if crop.size:
            crops.append(crop)
    return crops, threshold
