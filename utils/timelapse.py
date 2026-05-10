from __future__ import annotations

import os

import cv2
import nd2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class TimelapseND2ToTiffConverter:
    pixel_size_um = 0.108
    tiff_directory = "nd2totiff"
    processed_tiff_directory = "nd2totiff_processed"
    output_video_path = "timelapse_5fps.avi"
    position_index = 0

    @classmethod
    def convert_to_8bit(cls, array: np.ndarray) -> np.ndarray:
        if array.dtype == np.uint8:
            return array
        if np.issubdtype(array.dtype, np.integer):
            max_value = np.iinfo(array.dtype).max
            return (array.astype(np.float32) / max_value * 255).astype(np.uint8)
        return np.clip(array, 0, 255).astype(np.uint8)

    @classmethod
    def add_scale_bar(cls, image_path: str, scale_length_um: int = 10) -> Image.Image:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        width, height = img.size
        bar_width_pixels = scale_length_um / cls.pixel_size_um
        bar_height = 10
        bar_x = width - bar_width_pixels - 50
        bar_y = height - bar_height - 70
        draw.rectangle(
            [bar_x, bar_y, bar_x + bar_width_pixels, bar_y + bar_height], fill="white"
        )
        textsize = 50
        try:
            font = ImageFont.truetype("Arial Unicode.ttf", textsize)
        except OSError:
            font = ImageFont.load_default()
        text = f"{scale_length_um} um"
        text_width = draw.textlength(text, font=font)
        text_x = bar_x + (bar_width_pixels - text_width) / 2
        text_y = bar_y + bar_height - 10
        draw.text((text_x, text_y), text, fill="white", font=font)
        return img

    @classmethod
    def extract_nd2(cls, file_name: str) -> None:
        cls.prepare_directory(cls.tiff_directory)
        cls.prepare_directory(cls.processed_tiff_directory)

        with nd2.ND2File(file_name) as images:
            print(f"Available axes: {images.sizes.keys()}")
            print(f"Sizes: {images.sizes}")

            frame_indexes = [
                frame_index
                for frame_index, coords in enumerate(images.loop_indices)
                if coords.get("P", 0) == cls.position_index
            ]
            print(f"Using position index: {cls.position_index}")
            print(f"Frames: {len(frame_indexes)}")

            for n, frame_index in enumerate(frame_indexes):
                array = images.read_frame(frame_index)
                array = cls.convert_to_8bit(array)
                image = Image.fromarray(array)
                image.save(f"{cls.tiff_directory}/{n}.tif")

        for i in range(
            len([f for f in os.listdir(cls.tiff_directory) if f.endswith(".tif")])
        ):
            img = cls.add_scale_bar(f"{cls.tiff_directory}/{i}.tif", 10)
            img.save(f"{cls.processed_tiff_directory}/{i}.tif")
        cls.convert_to_video()

    @classmethod
    def prepare_directory(cls, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        for file_name in os.listdir(directory):
            if file_name.endswith(".tif"):
                os.remove(os.path.join(directory, file_name))

    @classmethod
    def convert_to_video(cls) -> None:
        tiff_files = [
            os.path.join(cls.processed_tiff_directory, f)
            for f in os.listdir(cls.processed_tiff_directory)
            if f.endswith(".tif")
        ]
        tiff_files = [
            f"{cls.processed_tiff_directory}/{n}.tif"
            for n in range(len(tiff_files))
            if n > 4
        ]
        if not tiff_files:
            raise ValueError("No processed TIFF files found for video conversion.")
        first_image = Image.open(tiff_files[0])
        frame_width, frame_height = first_image.size
        out = cv2.VideoWriter(
            cls.output_video_path,
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            5,
            (frame_width, frame_height),
        )
        for tiff_file in tiff_files:
            img = Image.open(tiff_file)
            img_array = np.array(img)
            if img_array.ndim == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            else:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            out.write(img_array)
        out.release()
