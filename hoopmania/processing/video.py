"""Video processing utilities."""

from pathlib import Path
from typing import Callable, Generator, Optional, Union
import subprocess

import numpy as np
import supervision as sv
from tqdm import tqdm


class VideoProcessor:
    """Video processing utilities wrapper."""

    def __init__(self, video_path: Union[str, Path]):
        """Initialize video processor.

        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)
        self._info = None

    @property
    def info(self) -> sv.VideoInfo:
        """Get video information."""
        if self._info is None:
            self._info = sv.VideoInfo.from_video_path(str(self.video_path))
        return self._info

    @property
    def fps(self) -> float:
        """Get video FPS."""
        return self.info.fps

    @property
    def width(self) -> int:
        """Get video width."""
        return self.info.width

    @property
    def height(self) -> int:
        """Get video height."""
        return self.info.height

    @property
    def total_frames(self) -> int:
        """Get total frame count."""
        return self.info.total_frames

    def get_frames(
        self,
        start: int = 0,
        end: Optional[int] = None,
        stride: int = 1,
    ) -> Generator[np.ndarray, None, None]:
        """Generate frames from video.

        Args:
            start: Starting frame index
            end: Ending frame index (exclusive)
            stride: Frame stride

        Yields:
            BGR frames as numpy arrays
        """
        return sv.get_video_frames_generator(
            source_path=str(self.video_path),
            start=start,
            end=end,
            stride=stride,
        )

    def process(
        self,
        output_path: Union[str, Path],
        callback: Callable[[np.ndarray, int], np.ndarray],
        show_progress: bool = True,
    ) -> Path:
        """Process video with callback function.

        Args:
            output_path: Path for output video
            callback: Function(frame, index) -> processed_frame
            show_progress: Whether to show progress bar

        Returns:
            Path to output video
        """
        output_path = Path(output_path)
        sv.process_video(
            source_path=str(self.video_path),
            target_path=str(output_path),
            callback=callback,
            show_progress=show_progress,
        )
        return output_path

    @staticmethod
    def compress(
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        crf: int = 28,
    ) -> Path:
        """Compress video using ffmpeg.

        Args:
            input_path: Path to input video
            output_path: Path for compressed video (default: adds -compressed suffix)
            crf: Constant Rate Factor (lower = higher quality)

        Returns:
            Path to compressed video
        """
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}-compressed{input_path.suffix}"
        else:
            output_path = Path(output_path)

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(input_path),
            "-vcodec", "libx264",
            "-crf", str(crf),
            str(output_path)
        ]
        subprocess.run(cmd, check=True)
        return output_path

    @staticmethod
    def create_sink(
        output_path: Union[str, Path],
        video_info: sv.VideoInfo,
    ) -> sv.VideoSink:
        """Create a video sink for writing frames.

        Args:
            output_path: Path for output video
            video_info: Video info for output

        Returns:
            VideoSink context manager
        """
        return sv.VideoSink(str(output_path), video_info)


def get_frame_at_index(
    video_path: Union[str, Path],
    index: int,
) -> np.ndarray:
    """Get a specific frame from video.

    Args:
        video_path: Path to video
        index: Frame index

    Returns:
        BGR frame as numpy array
    """
    generator = sv.get_video_frames_generator(
        source_path=str(video_path),
        start=index,
        iterative_seek=True,
    )
    return next(generator)
