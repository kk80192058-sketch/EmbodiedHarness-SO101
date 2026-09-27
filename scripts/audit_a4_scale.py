"""Offline A4 mapping audit against the independent printed 100 mm scale.

No camera or motor access. The ROI is chosen from a clear reference image,
independently of either homography being evaluated. Never edits calibration.
This checks a local planar mapping; it cannot establish robot alignment or Z.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


EXPECTED_MM = np.float32([[88.5, 158.0], [188.5, 158.0]])


def detect_scale_endpoints(image, roi):
    x1, y1, x2, y2 = roi
    if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
        raise ValueError('Scale ROI outside image')
    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(cv2.Canny(gray, 45, 100), 1, np.pi / 1800,
                            threshold=45, minLineLength=100, maxLineGap=8)
    if lines is None:
        raise ValueError('No continuous scale line found')
    candidates = []
    # OpenCV 4 commonly returns (N, 1, 4); OpenCV 5 may return (N, 4).
    # Normalise both public return layouts before interpreting a segment.
    for x0, y0, x3, y3 in np.asarray(lines).reshape(-1, 4):
        if abs(y3 - y0) < 100 or abs(x3 - x0) > 0.15 * abs(y3 - y0):
            continue
        first, last = sorted(((float(x0 + x1), float(y0 + y1)),
                              (float(x3 + x1), float(y3 + y1))), key=lambda p: p[1])
        candidates.append((np.linalg.norm(np.subtract(last, first)), first, last))
    if not candidates:
        raise ValueError('No unambiguous long scale line in expected orientation')
    candidates.sort(reverse=True)
    _, first, last = candidates[0]
    # The edges of one thin printed line may both be detected. A second distant
    # line with similar length is ambiguous and must not silently be selected.
    for length, other_first, _ in candidates[1:]:
        if length > candidates[0][0] * 0.8 and abs(other_first[0] - first[0]) > 6:
            raise ValueError('Multiple separate scale-like lines in ROI')
    return np.float32([first, last])


def evaluate(homography, endpoints, max_endpoint_error_mm=3.0):
    h = np.asarray(homography, dtype=float)
    if h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
        raise ValueError('Invalid homography')
    projected = cv2.perspectiveTransform(np.float32([endpoints]), h)[0]
    errors = np.linalg.norm(projected - EXPECTED_MM, axis=1)
    measured_length = float(np.linalg.norm(projected[1] - projected[0]))
    passed = bool(np.isfinite(projected).all() and max(errors) <= max_endpoint_error_mm
                  and abs(measured_length - 100) <= 3)
    return {'projected_endpoints_mm': projected.tolist(), 'expected_endpoints_mm': EXPECTED_MM.tolist(),
            'endpoint_errors_mm': errors.tolist(), 'measured_length_mm': measured_length,
            'maximum_endpoint_error_mm': max_endpoint_error_mm, 'scale_location_check_passed': passed,
            'motion_ready': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, action='append', required=True)
    parser.add_argument('--scale-roi', type=int, nargs=4, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error('Cannot read reference image')
    endpoints = detect_scale_endpoints(image, args.scale_roi)
    audits = []
    for path in args.calibration:
        calibration = json.loads(path.read_text())
        audits.append({'calibration': str(path), **evaluate(calibration['homography_pixel_to_board_mm'], endpoints)})
    result = {'mode': 'offline_independent_scale_audit', 'hardware_access': False,
              'source_image': str(args.image), 'roi_px': args.scale_roi,
              'detected_scale_endpoints_px': endpoints.tolist(),
              'expected_geometry_source': 'scripts/create_workspace_calibration_kit.py: draw_reference_board',
              'audits': audits,
              'limitations': ['Local planar scale check only; no robot-base alignment or gripper height validation.',
                             'ROI and printed orientation must correspond to the actual scale line.',
                             'Physical print-scale measurement is inherited from the existing user-confirmed setup, not measured by this tool.']}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    review = image.copy()
    cv2.rectangle(review, tuple(args.scale_roi[:2]), tuple(args.scale_roi[2:]), (255, 150, 0), 2)
    for label, point in zip(('scale 0', 'scale 100'), endpoints):
        location = tuple(np.int32(np.rint(point)))
        cv2.circle(review, location, 8, (0, 0, 255), 2)
        cv2.putText(review, label, (location[0] + 20, location[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
    if not cv2.imwrite(str(args.output_dir / 'scale_review.png'), review):
        raise RuntimeError('Could not save review image')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
