from __future__ import annotations

from utils import CellLengthAnalysisComponent, TimelapseND2ToTiffConverter


def main(
    file_name: str = "x60.nd2",
    celllength_image_path: str = "nd2totiff_processed/0.tif",
    celllength_canny_param_int: int = 85,
) -> None:
    TimelapseND2ToTiffConverter.extract_nd2(file_name)
    CellLengthAnalysisComponent.analyze_image(
        celllength_image_path,
        celllength_canny_param_int,
    )


if __name__ == "__main__":
    main()