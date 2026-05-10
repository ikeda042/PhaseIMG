from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, TypeAlias

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from pydantic.fields import Field
from tqdm import tqdm


CannyParamInt: TypeAlias = Annotated[int, Field(gt=1, lt=254)]
ImageArray: TypeAlias = np.ndarray
ContourList: TypeAlias = list[np.ndarray]
ImageShape: TypeAlias = tuple[int, ...]


class SearchCannyParamComponent:
    base_dir: Path = Path(__file__).resolve().parent.parent
    output_dir: str = "tmp"
    result_path: str = "result.gif"
    zoom_window_center_ratio: tuple[float, float] = (0.224, 0.399)
    zoom_window_size_ratio: float = 0.2
    zoom_window_color: tuple[int, int, int] = (0, 255, 255)
    scale_bar_length_um: float = 10.0
    scale_bar_pixel_size_um: float = 0.108
    scale_bar_right_margin_px: int = 50
    scale_bar_bottom_margin_px: int = 70
    scale_bar_height_px: int = 10
    scale_annotation_padding_px: int = 40

    @classmethod
    def get_contour(
        cls: type[SearchCannyParamComponent],
        image: ImageArray,
        canny_param_int: CannyParamInt,
    ) -> ContourList:
        gray: ImageArray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _threshold_value: float
        thresh: ImageArray
        _threshold_value, thresh = cv2.threshold(
            gray, canny_param_int, 255, cv2.THRESH_BINARY
        )
        thresh = cls.remove_scale_annotation(thresh)
        img_canny: ImageArray = cv2.Canny(thresh, 0, 150)
        contours_raw: tuple[np.ndarray, ...]
        hierarchy: ImageArray | None
        contours_raw, hierarchy = cv2.findContours(
            img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contours: ContourList = [c for c in contours_raw if cv2.contourArea(c) >= 3]
        return contours

    @classmethod
    def get_scale_annotation_region(
        cls: type[SearchCannyParamComponent],
        image_shape: ImageShape,
    ) -> tuple[int, int, int, int]:
        height: int = image_shape[0]
        width: int = image_shape[1]
        bar_width: int = int(
            np.ceil(cls.scale_bar_length_um / cls.scale_bar_pixel_size_um)
        )
        bar_x1: int = width - bar_width - cls.scale_bar_right_margin_px
        bar_y1: int = height - cls.scale_bar_height_px - cls.scale_bar_bottom_margin_px
        x1: int = max(0, bar_x1 - cls.scale_annotation_padding_px)
        y1: int = max(0, bar_y1 - cls.scale_annotation_padding_px)
        return x1, y1, width, height

    @classmethod
    def remove_scale_annotation(
        cls: type[SearchCannyParamComponent],
        image: ImageArray,
    ) -> ImageArray:
        x1: int
        y1: int
        x2: int
        y2: int
        x1, y1, x2, y2 = cls.get_scale_annotation_region(image.shape)
        image_without_annotation: ImageArray = image.copy()
        image_without_annotation[y1:y2, x1:x2] = 0
        return image_without_annotation

    @classmethod
    def plot_contour(
        cls: type[SearchCannyParamComponent],
        contour: ContourList,
        save_path: Path,
        number: int,
        image_shape: ImageShape,
    ) -> None:
        height: int = image_shape[0]
        width: int = image_shape[1]
        plt.figure()
        i: int
        for i in range(len(contour)):
            contour_i: ImageArray = contour[i].reshape(-1, 2).T
            plt.plot(contour_i[0], contour_i[1], linewidth=1)
        plt.gca().set_aspect("equal")
        plt.tick_params(axis="both", which="both", direction="in")
        plt.xlim(0, width)
        plt.ylim(height, 0)
        plt.text(
            0.95,
            0.95,
            str(number),
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        plt.savefig(save_path, dpi=300)
        plt.close()
        plt.clf()

    @classmethod
    def process_image(
        cls: type[SearchCannyParamComponent],
        image: ImageArray,
        canny_param_int: CannyParamInt,
    ) -> tuple[CannyParamInt, ContourList, float]:
        contours: ContourList = cls.get_contour(image, canny_param_int)
        contour_sum: float = sum([cv2.contourArea(c) for c in contours])
        return canny_param_int, contours, contour_sum

    @classmethod
    def plot_contour_sum(
        cls: type[SearchCannyParamComponent],
        global_contour_sums: list[float],
        i: int,
        output_dir: Path,
        area_axis_max: float,
    ) -> None:
        plt.figure()
        plt.plot(range(1, i + 1), global_contour_sums[:i], marker="o", markersize=1)
        plt.xlabel("Canny Threshold")
        plt.ylabel("Area")
        plt.xlim(-1, 254)
        plt.ylim(0, area_axis_max)
        plt.savefig(output_dir / f"contour_sum_{i}.png", dpi=300)
        plt.close()
        plt.clf()

    @classmethod
    def plot_contour_overlay(
        cls: type[SearchCannyParamComponent],
        image: ImageArray,
        contour: ContourList,
        save_path: Path,
        number: int,
    ) -> None:
        overlay: ImageArray = image.copy()
        cv2.drawContours(overlay, contour, -1, (0, 255, 0), 1)
        cv2.putText(
            overlay,
            f"Canny: {number}",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(save_path), overlay)

    @classmethod
    def resize_to_height(
        cls: type[SearchCannyParamComponent],
        image: ImageArray,
        target_height: int,
    ) -> ImageArray:
        height: int = image.shape[0]
        width: int = image.shape[1]
        if height == target_height:
            return image
        target_width: int = int(width * target_height / height)
        return cv2.resize(image, (target_width, target_height))

    @classmethod
    def get_area_axis_max(
        cls: type[SearchCannyParamComponent], contour_sums: list[float]
    ) -> float:
        max_area: float = max(contour_sums) if contour_sums else 1.0
        return max(max_area * 1.05, 1.0)

    @classmethod
    def get_zoom_window(
        cls: type[SearchCannyParamComponent], image: ImageArray
    ) -> tuple[int, int, int, int]:
        height: int = image.shape[0]
        width: int = image.shape[1]
        window_size: int = max(1, int(min(width, height) * cls.zoom_window_size_ratio))
        window_width: int = min(width, window_size)
        window_height: int = min(height, window_size)
        center_x: int = int(width * cls.zoom_window_center_ratio[0])
        center_y: int = int(height * cls.zoom_window_center_ratio[1])
        x1: int = min(max(center_x - window_width // 2, 0), width - window_width)
        y1: int = min(max(center_y - window_height // 2, 0), height - window_height)
        return x1, y1, x1 + window_width, y1 + window_height

    @classmethod
    def draw_zoom_window(
        cls: type[SearchCannyParamComponent],
        image: ImageArray,
        zoom_window: tuple[int, int, int, int],
    ) -> ImageArray:
        x1: int
        y1: int
        x2: int
        y2: int
        x1, y1, x2, y2 = zoom_window
        image_with_window: ImageArray = image.copy()
        cv2.rectangle(
            image_with_window,
            (x1, y1),
            (x2 - 1, y2 - 1),
            cls.zoom_window_color,
            4,
        )
        return image_with_window

    @classmethod
    def fit_to_panel(
        cls: type[SearchCannyParamComponent],
        image: ImageArray,
        target_width: int,
        target_height: int,
    ) -> ImageArray:
        height: int = image.shape[0]
        width: int = image.shape[1]
        scale: float = min(target_width / width, target_height / height)
        resized_width: int = max(1, int(width * scale))
        resized_height: int = max(1, int(height * scale))
        resized: ImageArray = cv2.resize(image, (resized_width, resized_height))
        panel: ImageArray = np.full(
            (target_height, target_width, 3), 255, dtype=np.uint8
        )
        x_offset: int = (target_width - resized_width) // 2
        y_offset: int = (target_height - resized_height) // 2
        panel[
            y_offset : y_offset + resized_height,
            x_offset : x_offset + resized_width,
        ] = resized
        return panel

    @classmethod
    def create_zoom_panel(
        cls: type[SearchCannyParamComponent],
        image: ImageArray,
        zoom_window: tuple[int, int, int, int],
        target_width: int,
        target_height: int,
    ) -> ImageArray:
        x1: int
        y1: int
        x2: int
        y2: int
        x1, y1, x2, y2 = zoom_window
        crop: ImageArray = image[y1:y2, x1:x2]
        return cls.fit_to_panel(crop, target_width, target_height)

    @classmethod
    def combine_panels(
        cls: type[SearchCannyParamComponent],
        contour_image: ImageArray,
        contour_sum_image: ImageArray,
        contour_overlay_image: ImageArray,
    ) -> ImageArray:
        panel_height: int = contour_image.shape[0]
        panel_width: int = contour_image.shape[1]
        zoom_window: tuple[int, int, int, int] = cls.get_zoom_window(
            contour_overlay_image
        )
        overlay_with_window: ImageArray = cls.draw_zoom_window(
            contour_overlay_image, zoom_window
        )
        panel1: ImageArray = cls.fit_to_panel(
            contour_image, panel_width, panel_height
        )
        panel2: ImageArray = cls.fit_to_panel(
            contour_sum_image, panel_width, panel_height
        )
        panel3: ImageArray = cls.fit_to_panel(
            overlay_with_window, panel_width, panel_height
        )
        panel4: ImageArray = cls.create_zoom_panel(
            contour_overlay_image, zoom_window, panel_width, panel_height
        )
        top_row: ImageArray = np.concatenate([panel1, panel2], axis=1)
        bottom_row: ImageArray = np.concatenate([panel3, panel4], axis=1)
        return np.concatenate([top_row, bottom_row], axis=0)

    @classmethod
    def get_project_path(
        cls: type[SearchCannyParamComponent], path: str | Path
    ) -> Path:
        resolved_path: Path = Path(path)
        if resolved_path.is_absolute():
            return resolved_path
        return cls.base_dir / resolved_path

    @classmethod
    def resolve_image_path(cls: type[SearchCannyParamComponent], tif_path: str) -> Path:
        path: Path = cls.get_project_path(tif_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Image file does not exist: {path}")

    @classmethod
    def load_image(
        cls: type[SearchCannyParamComponent], image_path: Path
    ) -> ImageArray:
        image: ImageArray | None = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image file: {image_path}")
        return image

    @classmethod
    def prepare_output_dir(
        cls: type[SearchCannyParamComponent], output_dir: Path
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        patterns: tuple[str, ...] = (
            "contour_*.png",
            "contour_sum_*.png",
            "contour_overlay_*.png",
            "combined_*.png",
        )
        pattern: str
        path: Path
        for pattern in patterns:
            for path in output_dir.glob(pattern):
                path.unlink()

    @classmethod
    def search_canny_param(
        cls: type[SearchCannyParamComponent],
        tif_path: str,
        output_dir: str | Path | None = None,
        result_path: str | Path | None = None,
    ) -> None:
        image_path: Path = cls.resolve_image_path(tif_path)
        output_path: Path = cls.get_project_path(output_dir or cls.output_dir)
        result_output_path: Path = cls.get_project_path(result_path or cls.result_path)
        cls.prepare_output_dir(output_path)

        print(f"Using image: {image_path}")
        image: ImageArray = cls.load_image(image_path)
        global_contour_sums: list[float] = []

        canny_param_int: int
        for canny_param_int in tqdm(range(1, 255)):
            processed_canny_param_int: CannyParamInt
            contours: ContourList
            contour_sum: float
            processed_canny_param_int, contours, contour_sum = cls.process_image(
                image, canny_param_int
            )
            global_contour_sums.append(contour_sum)
            cls.plot_contour(
                contours,
                output_path / f"contour_{processed_canny_param_int}.png",
                processed_canny_param_int,
                image.shape,
            )
            cls.plot_contour_overlay(
                image,
                contours,
                output_path / f"contour_overlay_{processed_canny_param_int}.png",
                processed_canny_param_int,
            )

        area_axis_max: float = cls.get_area_axis_max(global_contour_sums)
        for i in tqdm(range(1, 255)):
            cls.plot_contour_sum(global_contour_sums, i, output_path, area_axis_max)

        i: int
        for i in tqdm(range(1, 255)):
            img1: ImageArray | None = cv2.imread(str(output_path / f"contour_{i}.png"))
            img2: ImageArray | None = cv2.imread(
                str(output_path / f"contour_sum_{i}.png")
            )
            img3: ImageArray | None = cv2.imread(
                str(output_path / f"contour_overlay_{i}.png")
            )
            if img1 is None or img2 is None or img3 is None:
                raise ValueError(f"Could not read generated plot for threshold {i}")
            combined: ImageArray = cls.combine_panels(img1, img2, img3)
            cv2.imwrite(str(output_path / f"combined_{i}.png"), combined)

        writer: Any
        gif_frame_indexes: list[int] = [*range(50, 255), *range(1, 50)]
        temporary_result_output_path: Path = result_output_path.with_name(
            f".{result_output_path.stem}.tmp{result_output_path.suffix}"
        )
        temporary_result_output_path.unlink(missing_ok=True)
        with imageio.get_writer(
            temporary_result_output_path, mode="I", loop=0
        ) as writer:
            for i in tqdm(gif_frame_indexes):
                frame: ImageArray = imageio.imread(output_path / f"combined_{i}.png")
                writer.append_data(frame)
        temporary_result_output_path.replace(result_output_path)


def main(tif_path: str) -> None:
    SearchCannyParamComponent.search_canny_param(tif_path)


if __name__ == "__main__":
    main("nd2totiff_processed/0.tif")
