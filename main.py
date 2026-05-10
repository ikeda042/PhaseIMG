from __future__ import annotations

from utils import TimelapseND2ToTiffConverter


def main(file_name: str = "x60.nd2") -> None:
    TimelapseND2ToTiffConverter.extract_nd2(file_name)


if __name__ == "__main__":
    main()
