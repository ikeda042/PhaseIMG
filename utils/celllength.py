from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .search_canny_param_component import SearchCannyParamComponent


ContourLike = Sequence[Sequence[float]] | np.ndarray
CellLengthResult = dict[str, str | int | float]


@dataclass(frozen=True)
class CellLengthCalculator:
    """Calculate cell length in pixels from contour coordinates.

    The default algorithm mirrors the PhenoPixel6.0 centerline flow used for
    contour normalization:

    1. rasterize the contour and collect pixels inside the cell,
    2. convert those pixels into the PCA-aligned coordinate system,
    3. fit the centerline with a polynomial,
    4. integrate sqrt(1 + f'(x)^2) along the fitted centerline.
    """

    min_points: int = 2
    degree: int = 4
    arc_steps: int = 2048
    mask_padding: int = 2

    def calculate(self, contour: ContourLike) -> float:
        points = self._as_points(contour)
        if points.shape[0] < self.min_points:
            return 0.0

        shifted_points, coords_inside_cell, x_idx, y_idx = self._rasterize_contour(
            points
        )
        if x_idx.size == 0:
            return 0.0

        X = np.vstack((x_idx, y_idx))
        (
            _u1,
            _u2,
            _u1_contour,
            _u2_contour,
            min_u1,
            max_u1,
            _u1_c,
            _u2_c,
            U,
            _contour_U,
        ) = self._basis_conversion(
            shifted_points,
            X,
            coords_inside_cell,
        )

        coefficients = self._poly_fit(U)
        if coefficients.size == 0:
            return 0.0

        endpoint_min_u1, endpoint_max_u1 = self._centerline_endpoint_range(
            coefficients,
            _contour_U,
            float(min_u1),
            float(max_u1),
        )
        _xs, arc_lengths = self._build_arc_length_lookup(
            coefficients,
            endpoint_min_u1,
            endpoint_max_u1,
        )
        return float(arc_lengths[-1]) if arc_lengths.size > 0 else 0.0

    def calculate_pca_length(self, contour: ContourLike) -> float:
        points = self._as_points(contour)
        if points.shape[0] < self.min_points:
            return 0.0

        centered = points - points.mean(axis=0)
        if np.allclose(centered, 0.0):
            return 0.0

        axis = self._principal_axis(centered)
        projections = centered @ axis
        return float(projections.max() - projections.min())

    def __call__(self, contour: ContourLike) -> float:
        return self.calculate(contour)

    @staticmethod
    def _as_points(contour: ContourLike) -> np.ndarray:
        arr = np.asarray(contour, dtype=float)

        if arr.ndim == 3 and arr.shape[1:] == (1, 2):
            arr = arr[:, 0, :]
        elif arr.ndim == 2 and arr.shape[1] == 2:
            pass
        else:
            raise ValueError(
                "Contour must be shaped like (N, 2) or OpenCV-style (N, 1, 2)."
            )

        finite_rows = np.isfinite(arr).all(axis=1)
        return arr[finite_rows]

    def _rasterize_contour(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        min_xy = np.floor(points.min(axis=0)).astype(int)
        max_xy = np.ceil(points.max(axis=0)).astype(int)
        size_xy = max_xy - min_xy + 1 + self.mask_padding * 2
        if np.any(size_xy <= 0):
            empty = np.array([], dtype=int)
            return points.copy(), np.empty((0, 2), dtype=float), empty, empty

        shifted = points - min_xy + self.mask_padding
        contour_for_mask = np.rint(shifted).astype(np.int32).reshape(-1, 1, 2)
        height = int(size_xy[1])
        width = int(size_xy[0])
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [contour_for_mask], 255)

        y_idx, x_idx = np.nonzero(mask)
        coords_inside_cell = np.column_stack((y_idx, x_idx)).astype(float)
        return shifted, coords_inside_cell, x_idx.astype(float), y_idx.astype(float)

    @staticmethod
    def _principal_axis(centered_points: np.ndarray) -> np.ndarray:
        cov = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        norm = np.linalg.norm(axis)
        if norm == 0:
            return np.array([1.0, 0.0], dtype=float)
        return axis / norm

    def _basis_conversion(
        self,
        contour: np.ndarray,
        X: np.ndarray,
        coordinates_inside_cell: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        float,
        float,
        float,
        np.ndarray,
        np.ndarray,
    ]:
        coords_arr = np.asarray(coordinates_inside_cell, dtype=float).reshape(-1, 2)
        contour_arr = np.asarray(contour, dtype=float).reshape(-1, 2)
        center_arr = np.array(
            [
                float(coords_arr[:, 1].max() + coords_arr[:, 1].min()) / 2.0,
                float(coords_arr[:, 0].max() + coords_arr[:, 0].min()) / 2.0,
            ]
        )

        sigma = np.cov(X)
        eigenvalues, eigenvectors = np.linalg.eig(sigma)

        if eigenvalues[1] < eigenvalues[0]:
            Q = np.array([eigenvectors[1], eigenvectors[0]])
            U = (coords_arr @ Q)[:, ::-1]
            contour_U = (contour_arr[:, ::-1] @ Q)[:, ::-1]
            u1_c, u2_c = center_arr @ Q
        else:
            Q = np.array([eigenvectors[0], eigenvectors[1]])
            U = coords_arr[:, ::-1] @ Q
            contour_U = contour_arr @ Q
            u2_c, u1_c = center_arr @ Q

        u1 = U[:, 1]
        u2 = U[:, 0]
        u1_contour = contour_U[:, 1]
        u2_contour = contour_U[:, 0]
        return (
            u1,
            u2,
            u1_contour,
            u2_contour,
            float(u1.min()),
            float(u1.max()),
            float(u1_c),
            float(u2_c),
            U,
            contour_U,
        )

    def _poly_fit(self, values: np.ndarray) -> np.ndarray:
        values_arr = np.asarray(values, dtype=float)
        if values_arr.size == 0:
            return np.array([])

        degree = min(self.degree, max(0, values_arr.shape[0] - 1))
        u1_values = values_arr[:, 1]
        f_values = values_arr[:, 0]
        W = np.vander(u1_values, degree + 1)
        try:
            coefficients = np.linalg.inv(W.T @ W) @ W.T @ f_values
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(W) @ f_values
        return coefficients

    def _build_arc_length_lookup(
        self,
        coefficients: np.ndarray,
        min_x: float,
        max_x: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(min_x) or not np.isfinite(max_x) or max_x <= min_x:
            return np.array([min_x, max_x], dtype=float), np.array(
                [0.0, 0.0], dtype=float
            )

        poly = np.poly1d(coefficients)
        poly_der = np.polyder(poly)
        xs = np.linspace(min_x, max_x, self.arc_steps)
        slopes = poly_der(xs)
        integrand = np.sqrt(1.0 + slopes * slopes)
        cumulative = np.zeros_like(xs)
        if xs.size > 1:
            dx = np.diff(xs)
            cumulative[1:] = np.cumsum((integrand[1:] + integrand[:-1]) * 0.5 * dx)
        return xs, cumulative

    def _centerline_endpoint_range(
        self,
        coefficients: np.ndarray,
        contour_U: np.ndarray,
        min_u1: float,
        max_u1: float,
    ) -> tuple[float, float]:
        projected_u1, _projected_u2, _distances = self._project_points_to_polynomial(
            coefficients,
            np.asarray(contour_U[:, 1], dtype=float),
            np.asarray(contour_U[:, 0], dtype=float),
            min_u1,
            max_u1,
        )
        if projected_u1.size == 0:
            return min_u1, max_u1
        endpoint_min = float(projected_u1.min())
        endpoint_max = float(projected_u1.max())
        if not np.isfinite(endpoint_min) or not np.isfinite(endpoint_max):
            return min_u1, max_u1
        return endpoint_min, endpoint_max

    @staticmethod
    def _project_points_to_polynomial(
        coefficients: np.ndarray,
        x_values: np.ndarray,
        y_values: np.ndarray,
        min_x: float,
        max_x: float,
        *,
        iterations: int = 8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if coefficients.size == 0:
            empty = np.array([], dtype=float)
            return empty, empty, empty
        if not np.isfinite(min_x) or not np.isfinite(max_x) or max_x <= min_x:
            projected_x = np.asarray(x_values, dtype=float)
            projected_y = np.polyval(coefficients, projected_x)
            distances = np.abs(projected_y - np.asarray(y_values, dtype=float))
            return projected_x, projected_y, distances

        x_q = np.asarray(x_values, dtype=float)
        y_q = np.asarray(y_values, dtype=float)
        x = np.clip(x_q, min_x, max_x)

        poly = np.poly1d(coefficients)
        poly_der = np.polyder(poly)
        poly_second = np.polyder(poly_der)

        for _ in range(iterations):
            y = poly(x)
            dy = poly_der(x)
            ddy = poly_second(x)
            grad = 2.0 * (x - x_q) + 2.0 * (y - y_q) * dy
            hess = 2.0 + 2.0 * dy * dy + 2.0 * (y - y_q) * ddy
            step = np.divide(
                grad,
                hess,
                out=np.zeros_like(grad, dtype=float),
                where=np.abs(hess) > 1e-12,
            )
            next_x = np.clip(x - step, min_x, max_x)
            if np.max(np.abs(next_x - x), initial=0.0) < 1e-3:
                x = next_x
                break
            x = next_x
        x = np.where(np.isfinite(x), x, np.clip(x_q, min_x, max_x))

        candidates = (
            x,
            np.full_like(x, min_x, dtype=float),
            np.full_like(x, max_x, dtype=float),
        )
        best_x = candidates[0].copy()
        best_y = poly(best_x)
        best_dist_sq = (best_x - x_q) ** 2 + (best_y - y_q) ** 2
        for candidate_x in candidates[1:]:
            candidate_y = poly(candidate_x)
            dist_sq = (candidate_x - x_q) ** 2 + (candidate_y - y_q) ** 2
            better = dist_sq < best_dist_sq
            best_x[better] = candidate_x[better]
            best_y[better] = candidate_y[better]
            best_dist_sq[better] = dist_sq[better]
        distances = np.sqrt(best_dist_sq)
        return best_x, best_y, distances


def calculate_cell_length_px(contour: ContourLike) -> float:
    return CellLengthCalculator().calculate(contour)


class CellLengthAnalysisComponent:
    base_dir: Path = Path(__file__).resolve().parent.parent
    output_dir: str = "celllength_results"
    default_image_path: str = "nd2totiff_processed/0.tif"
    default_canny_param_int: int = 85
    csv_columns: tuple[str, ...] = (
        "image",
        "canny_param",
        "contour_index",
        "area_px2",
        "perimeter_px",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "poly_arc_length_px",
        "pca_length_px",
    )

    @classmethod
    def get_project_path(cls, path: str | Path) -> Path:
        resolved_path = Path(path)
        if resolved_path.is_absolute():
            return resolved_path
        return cls.base_dir / resolved_path

    @classmethod
    def get_frame_label(cls, image_path: str | Path) -> str:
        stem = Path(image_path).stem
        return f"frame{stem}" if stem.isdigit() else stem

    @classmethod
    def build_output_paths(
        cls,
        image_path: str | Path,
        canny_param_int: int,
        output_dir: str | Path | None = None,
    ) -> tuple[Path, Path, Path]:
        output_path = cls.get_project_path(output_dir or cls.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        frame_label = cls.get_frame_label(image_path)
        prefix = f"celllength_canny{canny_param_int}_{frame_label}"
        return (
            output_path / f"{prefix}.csv",
            output_path / f"{prefix}.json",
            output_path / f"{prefix}_overlay.png",
        )

    @classmethod
    def calculate_results(
        cls,
        image_path: str | Path,
        canny_param_int: int,
    ) -> tuple[list[CellLengthResult], np.ndarray, list[np.ndarray]]:
        project_image_path = cls.get_project_path(image_path)
        image = SearchCannyParamComponent.load_image(project_image_path)
        contours = SearchCannyParamComponent.get_contour(image, canny_param_int)
        calculator = CellLengthCalculator()
        rows: list[CellLengthResult] = []

        for contour_index, contour in enumerate(contours):
            bbox_x: int
            bbox_y: int
            bbox_w: int
            bbox_h: int
            bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(contour)
            rows.append(
                {
                    "image": str(Path(image_path)),
                    "canny_param": canny_param_int,
                    "contour_index": contour_index,
                    "area_px2": float(cv2.contourArea(contour)),
                    "perimeter_px": float(cv2.arcLength(contour, True)),
                    "bbox_x": int(bbox_x),
                    "bbox_y": int(bbox_y),
                    "bbox_w": int(bbox_w),
                    "bbox_h": int(bbox_h),
                    "poly_arc_length_px": calculator.calculate(contour),
                    "pca_length_px": calculator.calculate_pca_length(contour),
                }
            )

        return rows, image, contours

    @classmethod
    def write_csv(cls, csv_path: Path, rows: list[CellLengthResult]) -> None:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cls.csv_columns)
            writer.writeheader()
            writer.writerows(rows)

    @classmethod
    def write_json(cls, json_path: Path, rows: list[CellLengthResult]) -> None:
        with json_path.open("w") as f:
            json.dump(rows, f, indent=2)

    @classmethod
    def write_overlay(
        cls,
        overlay_path: Path,
        image: np.ndarray,
        contours: list[np.ndarray],
        rows: list[CellLengthResult],
    ) -> None:
        overlay = image.copy()
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
        row: CellLengthResult
        for row in rows:
            label = f"{row['contour_index']}: {row['poly_arc_length_px']:.1f}px"
            cv2.putText(
                overlay,
                label,
                (int(row["bbox_x"]), max(0, int(row["bbox_y"]) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(overlay_path), overlay)

    @classmethod
    def analyze_image(
        cls,
        image_path: str | Path | None = None,
        canny_param_int: int | None = None,
        output_dir: str | Path | None = None,
    ) -> list[CellLengthResult]:
        target_image_path = image_path or cls.default_image_path
        target_canny_param_int = canny_param_int or cls.default_canny_param_int
        csv_path, json_path, overlay_path = cls.build_output_paths(
            target_image_path,
            target_canny_param_int,
            output_dir,
        )
        rows: list[CellLengthResult]
        image: np.ndarray
        contours: list[np.ndarray]
        rows, image, contours = cls.calculate_results(
            target_image_path, target_canny_param_int
        )
        cls.write_csv(csv_path, rows)
        cls.write_json(json_path, rows)
        cls.write_overlay(overlay_path, image, contours, rows)
        return rows

    @classmethod
    def run_default(cls) -> list[CellLengthResult]:
        return cls.analyze_image(
            cls.default_image_path,
            cls.default_canny_param_int,
            cls.output_dir,
        )
