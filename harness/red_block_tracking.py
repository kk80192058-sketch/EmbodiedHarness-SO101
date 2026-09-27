"""Scene-specific red-block observations, with explicit ambiguity/visibility.

ROIs describe the verified 2026-09-27 camera views. These image coordinates
are not robot coordinates and cannot establish height or grasp success.
"""
import cv2
import numpy as np

ROIS = {'wrist': (0.25, 0.35, 0.80, 0.98), 'global': (0.44, 0.45, 0.66, 0.93)}


def detect_red_block(image, role):
    height, width = image.shape[:2]
    roi = ROIS[role]
    x0, y0, x1, y1 = [round(v * s) for v, s in zip(roi, [width, height, width, height], strict=True)]
    hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 120, 55), (12, 255, 255)) | cv2.inRange(hsv, (170, 120, 55), (180, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < width * height * 0.00025:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if not 0.45 < w / h < 2.2 or area / max(hull_area, 1) < 0.8:
            continue
        moments = cv2.moments(contour)
        candidates.append({'center_px': [x0 + moments['m10'] / moments['m00'],
                                         y0 + moments['m01'] / moments['m00']],
                           'bbox_px': [x0 + x, y0 + y, w, h], 'area_px': area,
                           'touches_roi_boundary': x <= 1 or y <= 1 or x + w >= x1 - x0 - 1 or y + h >= y1 - y0 - 1})
    valid = len(candidates) == 1 and not candidates[0]['touches_roi_boundary']
    return {'status': 'visible_unique' if valid else 'unknown',
            'target': candidates[0] if valid else None, 'candidates': candidates,
            'roi_px': [x0, y0, x1, y1], 'grasp_status': 'unknown'}
