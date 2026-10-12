# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportOptionalOperand=false
"""
GUI-independent image preprocessing IO helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path
import re
from typing import Callable, Literal, Sequence

import cv2
import numpy as np


ProgressCallback = Callable[[str, int, int, str], None]


class ImagePreprocessingIOError(RuntimeError):
    """Raised for preprocessing IO planning or read failures."""


@dataclass(frozen=True)
class PlannedFrameTask:
    """One planned source frame and deterministic output target."""

    cam_idx: int
    input_kind: Literal["tiff", "cine"]
    source_path: Path
    output_path: Path
    output_relpath: str
    output_index: int
    cine_frame: int | None = None


@dataclass(frozen=True)
class DetectedRootCameraInput:
    """Detected camera input under a root directory."""

    camera_name: str
    cine_path: Path | None = None
    image_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class DetectedRootInput:
    """Detected input mode and ordered camera sources for --input-root."""

    root_dir: Path
    input_kind: Literal["cine", "tiff"]
    cameras: tuple[DetectedRootCameraInput, ...]


def _emit_progress(callback: ProgressCallback | None, stage: str, current: int, total: int, message: str) -> None:
    if callback is not None:
        callback(stage, current, total, message)


def _as_path(value: str | Path) -> Path:
    if isinstance(value, Path):
        return value
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    return Path(text)


def _normalize_stride(stride: int) -> int:
    stride = int(stride)
    if stride <= 0:
        raise ValueError("stride must be >= 1")
    return stride


def _normalize_count(count: int | None) -> int | None:
    if count is None:
        return None
    count = int(count)
    if count <= 0:
        raise ValueError("count must be >= 1 when provided")
    return count


def _natural_sort_key(value: str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", value.casefold())
    key: list[object] = []
    for part in parts:
        if not part:
            continue
        key.append(int(part) if part.isdigit() else part)
    return tuple(key)


def _sorted_natural(paths: Sequence[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: _natural_sort_key(path.name))


def _iter_camera_directories(root_dir: Path) -> list[Path]:
    return sorted((path for path in root_dir.iterdir() if path.is_dir()), key=lambda path: _natural_sort_key(path.name))


def _iter_direct_files(root_dir: Path, suffixes: tuple[str, ...]) -> list[Path]:
    normalized = tuple(suffix.casefold() for suffix in suffixes)
    matches = [path for path in root_dir.iterdir() if path.is_file() and path.suffix.casefold() in normalized]
    return sorted(matches, key=lambda path: _natural_sort_key(path.name))


def detect_input_root(root_dir: str | Path) -> DetectedRootInput:
    """Detect ordered camera inputs under a root directory for CLI root mode."""
    root_path = _as_path(root_dir).expanduser().resolve()
    if not root_path.exists():
        raise ImagePreprocessingIOError(f"Input root does not exist: {root_path}")
    if not root_path.is_dir():
        raise ImagePreprocessingIOError(f"Input root must be a directory: {root_path}")

    camera_dirs = _iter_camera_directories(root_path)

    cine_cameras: list[DetectedRootCameraInput] = []
    for camera_dir in camera_dirs:
        cine_files = _iter_direct_files(camera_dir, (".cine",))
        if not cine_files:
            continue
        cine_cameras.append(DetectedRootCameraInput(camera_name=camera_dir.name, cine_path=cine_files[0]))

    if cine_cameras:
        return DetectedRootInput(root_dir=root_path, input_kind="cine", cameras=tuple(cine_cameras))

    root_cines = _iter_direct_files(root_path, (".cine",))
    if root_cines:
        return DetectedRootInput(
            root_dir=root_path,
            input_kind="cine",
            cameras=tuple(
                DetectedRootCameraInput(camera_name=path.stem, cine_path=path)
                for path in root_cines
            ),
        )

    tiff_suffixes = (".tif", ".tiff")
    tiff_cameras: list[DetectedRootCameraInput] = []
    for camera_dir in camera_dirs:
        image_paths = tuple(_iter_direct_files(camera_dir, tiff_suffixes))
        if not image_paths:
            continue
        tiff_cameras.append(DetectedRootCameraInput(camera_name=camera_dir.name, image_paths=image_paths))

    if tiff_cameras:
        return DetectedRootInput(root_dir=root_path, input_kind="tiff", cameras=tuple(tiff_cameras))

    raise ImagePreprocessingIOError(
        "No supported camera inputs were found under the input root. "
        "Expected CINE files in ROOT/CamN/*.cine or ROOT/camN.cine, or TIFF files in camera subdirectories."
    )


def _to_gray_float32(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def _import_pycine():
    try:
        from pycine.file import read_header
        from pycine.raw import read_frames
    except Exception as exc:  # pragma: no cover - depends on optional runtime dependency
        raise ImagePreprocessingIOError(
            "CINE input requested but pycine is unavailable. Install pycine to enable CINE preprocessing."
        ) from exc
    return read_frames, read_header


def read_image_list_file(image_list_file: str | Path) -> list[Path]:
    """
    Read an image list text file and resolve entries relative to the list file.
    """
    list_path = _as_path(image_list_file).expanduser().resolve()
    parent = list_path.parent
    resolved_paths: list[Path] = []

    with list_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            entry = Path(line)
            if not entry.is_absolute():
                entry = parent / entry
            resolved_paths.append(entry.resolve())

    return resolved_paths


def resolve_tiff_input_paths(
    image_list_files: Sequence[str | Path] | None = None,
    image_paths: Sequence[str | Path] | None = None,
    *,
    base_dir: str | Path | None = None,
) -> list[Path]:
    """
    Resolve TIFF input paths from list files and/or direct paths.
    """
    resolved: list[Path] = []

    for image_list_file in image_list_files or []:
        resolved.extend(read_image_list_file(image_list_file))

    if image_paths:
        base_path = _as_path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd().resolve()
        for image_path in image_paths:
            entry = _as_path(image_path).expanduser()
            if not entry.is_absolute():
                entry = base_path / entry
            resolved.append(entry.resolve())

    return resolved


def _build_tiff_output_names(paths: Sequence[Path], naming: str, start_index: int) -> list[str]:
    if naming == "frame":
        return [f"frame_{start_index + idx:06d}.tif" for idx in range(len(paths))]

    if naming != "source":
        raise ValueError("naming must be 'frame' or 'source'")

    output_names: list[str] = []
    used_names: dict[str, int] = {}
    for idx, path in enumerate(paths):
        base_name = f"{path.stem}.tif"
        count = used_names.get(base_name, 0)
        used_names[base_name] = count + 1
        if count == 0:
            output_names.append(base_name)
        else:
            output_names.append(f"{path.stem}_{start_index + idx:06d}.tif")
    return output_names


def _build_cine_output_names(cine_path: Path, frames: Sequence[int], naming: str, start_index: int) -> list[str]:
    if naming == "frame":
        return [f"frame_{start_index + idx:06d}.tif" for idx in range(len(frames))]

    if naming != "source":
        raise ValueError("naming must be 'frame' or 'source'")

    stem = cine_path.stem
    return [f"{stem}_frame_{frame_idx:06d}.tif" for frame_idx in frames]


def plan_tiff_tasks(
    *,
    cam_idx: int,
    output_dir: str | Path,
    image_list_files: Sequence[str | Path] | None = None,
    image_paths: Sequence[str | Path] | None = None,
    base_dir: str | Path | None = None,
    naming: Literal["frame", "source"] = "frame",
    start_index: int = 0,
) -> list[PlannedFrameTask]:
    """
    Plan TIFF processing tasks from image lists and/or direct paths.
    """
    source_paths = resolve_tiff_input_paths(image_list_files, image_paths, base_dir=base_dir)
    if not source_paths:
        raise ValueError("No TIFF inputs were provided")

    output_root = _as_path(output_dir).expanduser().resolve()
    camera_dir = output_root / f"cam{cam_idx}"
    output_names = _build_tiff_output_names(source_paths, naming, int(start_index))

    tasks: list[PlannedFrameTask] = []
    for idx, (source_path, output_name) in enumerate(zip(source_paths, output_names)):
        output_path = camera_dir / output_name
        relpath = output_path.relative_to(output_root).as_posix()
        tasks.append(
            PlannedFrameTask(
                cam_idx=int(cam_idx),
                input_kind="tiff",
                source_path=source_path,
                output_path=output_path,
                output_relpath=relpath,
                output_index=int(start_index) + idx,
            )
        )
    return tasks


def get_cine_frame_range(cine_path: str | Path) -> tuple[int, int]:
    """
    Return inclusive first/last frame numbers for a CINE file.
    """
    _, read_header = _import_pycine()
    cine_path = _as_path(cine_path).expanduser().resolve()
    header = read_header(str(cine_path))
    cine_header = header["cinefileheader"]
    first_frame = int(cine_header.FirstImageNo)
    last_frame = first_frame + int(cine_header.ImageCount) - 1
    return first_frame, last_frame


def read_cine_frames(
    cine_path: str | Path,
    frame: int,
    count: int = 1,
    header: dict | None = None,
) -> list[np.ndarray]:
    """
    Read ``count`` consecutive CINE frames, starting at ``frame``.

    ``frame`` is in the camera's own numbering (FirstImageNo ... FirstImageNo + ImageCount - 1, as listed by
    get_cine_frame_range).  It must not be handed to pycine's ``read_frames(start_frame=...)`` as it is: pycine
    reads the stored frame at position ``start_frame - 1`` (counted from 1), takes 0 as 1, and takes a negative
    position from the end of the file.  Passing the frame number made every chunk after the first start one frame
    early (a copy of frame 255 at 256 for chunks of 256, and the last frame lost).  Here pycine gets the position
    ``frame - FirstImageNo + 1``, which is always 1 or more.
    """
    read_frames, read_header = _import_pycine()
    if header is None:
        header = read_header(str(cine_path))
    cine_header = header["cinefileheader"]
    first_no = int(cine_header.FirstImageNo)
    last_no = first_no + int(cine_header.ImageCount) - 1
    frame = int(frame)
    count = int(count)
    if count < 1:
        raise ValueError("count must be >= 1")
    if frame < first_no or frame + count - 1 > last_no:
        raise ValueError(
            f"Requested CINE frames {frame}-{frame + count - 1} are outside available range {first_no}-{last_no}"
        )
    raw_images, _, _ = read_frames(str(cine_path), start_frame=frame - first_no + 1, count=count)
    frames = [np.asarray(img) for img in raw_images]
    if len(frames) != count:
        raise ImagePreprocessingIOError(
            f"Read {len(frames)} of {count} frames from {cine_path} starting at frame {frame}"
        )
    return frames


def plan_cine_tasks(
    *,
    cam_idx: int,
    cine_path: str | Path,
    output_dir: str | Path,
    start_frame: int | None = None,
    end_frame: int | None = None,
    naming: Literal["frame", "source"] = "frame",
    start_index: int = 0,
) -> list[PlannedFrameTask]:
    """
    Plan CINE frame extraction tasks with inclusive frame range.
    """
    cine_path = _as_path(cine_path).expanduser().resolve()
    first_frame, last_frame = get_cine_frame_range(cine_path)

    start_frame = first_frame if start_frame is None else int(start_frame)
    end_frame = last_frame if end_frame is None else int(end_frame)

    if start_frame > end_frame:
        raise ValueError("start_frame must be <= end_frame")
    if start_frame < first_frame or end_frame > last_frame:
        raise ValueError(
            f"Requested CINE frame range {start_frame}-{end_frame} is outside available range {first_frame}-{last_frame}"
        )

    frames = list(range(start_frame, end_frame + 1))
    output_root = _as_path(output_dir).expanduser().resolve()
    camera_dir = output_root / f"cam{cam_idx}"
    output_names = _build_cine_output_names(cine_path, frames, naming, int(start_index))

    tasks: list[PlannedFrameTask] = []
    for idx, (frame_idx, output_name) in enumerate(zip(frames, output_names)):
        output_path = camera_dir / output_name
        relpath = output_path.relative_to(output_root).as_posix()
        tasks.append(
            PlannedFrameTask(
                cam_idx=int(cam_idx),
                input_kind="cine",
                source_path=cine_path,
                output_path=output_path,
                output_relpath=relpath,
                output_index=int(start_index) + idx,
                cine_frame=frame_idx,
            )
        )
    return tasks


def read_cine_frame(cine_path: str | Path, frame_index: int) -> np.ndarray:
    """
    Read one CINE frame (``frame_index`` in the camera's numbering) as a numpy array.
    """
    cine_path = _as_path(cine_path).expanduser().resolve()
    return read_cine_frames(cine_path, int(frame_index), count=1)[0]


def load_task_image(task: PlannedFrameTask) -> np.ndarray:
    """
    Load one planned task image from TIFF or CINE input.
    """
    if task.input_kind == "tiff":
        img = cv2.imread(str(task.source_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ImagePreprocessingIOError(f"Failed to read TIFF image: {task.source_path}")
        return img

    if task.input_kind == "cine":
        if task.cine_frame is None:
            raise ImagePreprocessingIOError("CINE task is missing cine_frame")
        return read_cine_frame(task.source_path, task.cine_frame)

    raise ImagePreprocessingIOError(f"Unsupported task input kind: {task.input_kind}")


def _iter_selected_tiff_paths(paths: Sequence[Path], start: int, count: int | None, stride: int):
    start = max(0, int(start))
    stride = _normalize_stride(stride)
    count = _normalize_count(count)
    selected = islice(range(start, len(paths), stride), 0, count)
    for index in selected:
        yield index, paths[index]


def compute_mean_background_from_tiff(
    *,
    image_list_files: Sequence[str | Path] | None = None,
    image_paths: Sequence[str | Path] | None = None,
    base_dir: str | Path | None = None,
    start: int = 0,
    count: int | None = None,
    stride: int = 1,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """
    Compute a streaming mean background from TIFF inputs.
    """
    paths = resolve_tiff_input_paths(image_list_files, image_paths, base_dir=base_dir)
    if not paths:
        raise ValueError("No TIFF inputs were provided for background calculation")

    selected = list(_iter_selected_tiff_paths(paths, start, count, stride))
    if not selected:
        raise ValueError("No TIFF frames selected for background calculation")

    accumulator = None
    total = len(selected)

    for current, (_, path) in enumerate(selected, start=1):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ImagePreprocessingIOError(f"Failed to read TIFF image for background: {path}")

        gray = _to_gray_float32(img)
        if accumulator is None:
            accumulator = gray.astype(np.float64)
        else:
            accumulator += gray.astype(np.float64)

        _emit_progress(progress_callback, "background", current, total, f"Background {current}/{total}")

    return (accumulator / total).astype(np.float32)


def compute_mean_background_from_cine(
    *,
    cine_path: str | Path,
    start: int = 0,
    count: int | None = None,
    stride: int = 1,
    chunk_size: int = 128,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """
    Compute a streaming mean background from CINE frames.
    """
    _, read_header = _import_pycine()
    cine_path = _as_path(cine_path).expanduser().resolve()
    header = read_header(str(cine_path))
    first_frame, last_frame = get_cine_frame_range(cine_path)
    stride = _normalize_stride(stride)
    count = _normalize_count(count)
    start_frame = max(int(start), first_frame)

    frame_numbers = list(islice(range(start_frame, last_frame + 1, stride), 0, count))
    if not frame_numbers:
        raise ValueError("No CINE frames selected for background calculation")

    accumulator = None
    total = len(frame_numbers)
    processed = 0
    chunk_size = max(1, int(chunk_size))

    if stride == 1:
        batch_start = 0
        while batch_start < total:
            batch_frames = frame_numbers[batch_start:batch_start + chunk_size]
            first = batch_frames[0]
            raw_images = read_cine_frames(cine_path, first, count=len(batch_frames), header=header)
            for offset, raw in enumerate(raw_images):
                gray = _to_gray_float32(np.asarray(raw))
                if accumulator is None:
                    accumulator = gray.astype(np.float64)
                else:
                    accumulator += gray.astype(np.float64)
                processed += 1
                _emit_progress(progress_callback, "background", processed, total, f"Background {processed}/{total}")
            batch_start += len(batch_frames)
    else:
        for frame_index in frame_numbers:
            frames = read_cine_frames(cine_path, int(frame_index), count=1, header=header)
            gray = _to_gray_float32(np.asarray(frames[0]))
            if accumulator is None:
                accumulator = gray.astype(np.float64)
            else:
                accumulator += gray.astype(np.float64)
            processed += 1
            _emit_progress(progress_callback, "background", processed, total, f"Background {processed}/{total}")

    return (accumulator / total).astype(np.float32)


def write_image_list_file(image_list_path: str | Path, relative_paths: Sequence[str]) -> Path:
    """
    Write a camera image list file with OpenLPT-friendly relative paths.
    """
    image_list_path = _as_path(image_list_path).expanduser().resolve()
    image_list_path.parent.mkdir(parents=True, exist_ok=True)
    with image_list_path.open("w", encoding="utf-8", newline="\n") as handle:
        for relpath in relative_paths:
            handle.write(f"{Path(relpath).as_posix()}\n")
    return image_list_path
