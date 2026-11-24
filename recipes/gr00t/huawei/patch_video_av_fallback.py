#!/usr/bin/env python3
"""Patch gr00t video_utils to fall back to PyAV when torchcodec is unavailable (Ascend)."""
from pathlib import Path

p = Path("/home/gr00t/src/Isaac-GR00T/gr00t/utils/video_utils.py")
text = p.read_text()
if "GR00T_AV_FALLBACK" in text:
    print("ALREADY_PATCHED")
    raise SystemExit(0)

# Replace _get_video_decoder_cls / _build_decoder / get_frames_* with av-aware versions
# Keep get_accumulate_timestamp_idxs untouched.

marker = "def _get_video_decoder_cls():"
start = text.find(marker)
if start < 0:
    raise SystemExit("NO_DECODER")
# Keep everything from get_accumulate_timestamp_idxs onward
acc = text.find("\ndef get_accumulate_timestamp_idxs(")
if acc < 0:
    raise SystemExit("NO_ACC")

head = text[:start]
tail = text[acc:]

new_mid = r'''
# GR00T_AV_FALLBACK — Ascend has no CUDA NVRTC; torchcodec wheels need it.
# Prefer torchcodec when it loads; otherwise decode with PyAV.

def _torchcodec_available() -> bool:
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
        return True
    except Exception:
        return False


def _get_video_decoder_cls():
    try:
        from torchcodec.decoders import VideoDecoder
        return VideoDecoder
    except (ImportError, OSError, RuntimeError) as exc:
        raise _decoder_import_error(exc) from exc


def _build_decoder(video_path: str, decoder_kwargs: Optional[dict]):
    video_decoder_cls = _get_video_decoder_cls()
    kwargs = {**_DEFAULT_DECODER_KWARGS, **(decoder_kwargs or {})}
    return video_decoder_cls(video_path, **kwargs)


def _av_frames_at_indices(video_path: str, indices: list[int] | np.ndarray) -> np.ndarray:
    import av

    indices = np.asarray(indices, dtype=np.int64)
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        wanted = set(int(i) for i in indices.tolist())
        max_idx = int(indices.max()) if len(indices) else -1
        by_idx: dict[int, np.ndarray] = {}
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted:
                by_idx[i] = frame.to_ndarray(format="rgb24")
            if i >= max_idx and len(by_idx) == len(wanted):
                break
    missing = [i for i in indices.tolist() if int(i) not in by_idx]
    if missing:
        raise RuntimeError(f"PyAV could not decode frames {missing[:8]}... from {video_path}")
    return np.stack([by_idx[int(i)] for i in indices.tolist()], axis=0)


def _av_frames_at_timestamps(
    video_path: str, timestamps: list[float] | np.ndarray
) -> np.ndarray:
    import av

    timestamps = np.asarray(timestamps, dtype=np.float64)
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate) if stream.average_rate else 15.0
        indices = np.clip(np.round(timestamps * fps).astype(np.int64), 0, None)
    return _av_frames_at_indices(video_path, indices)


def get_frames_by_indices(
    video_path: str,
    indices: list[int] | np.ndarray,
    decoder_kwargs: Optional[dict] = None,
) -> np.ndarray:
    if _torchcodec_available():
        decoder = _build_decoder(video_path, decoder_kwargs)
        return decoder.get_frames_at(indices=indices).data.numpy()
    return _av_frames_at_indices(video_path, indices)


def get_frames_by_timestamps(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    decoder_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """Get frames from a video at specified timestamps."""
    if not _torchcodec_available():
        return _av_frames_at_timestamps(video_path, timestamps)

    decoder = _build_decoder(video_path, decoder_kwargs)
    fps = decoder.metadata.average_fps
    interval = 1 / fps
    timestamps = np.array(timestamps).astype(np.float64)
    closest_timestamps = np.round(timestamps / interval) * interval
    timestamp_errors = np.abs(closest_timestamps - timestamps) / interval
    invalid_mask = timestamp_errors >= 0.01
    if np.any(invalid_mask):
        invalid_indices = np.where(invalid_mask)[0]
        invalid_timestamps = timestamps[invalid_indices]
        raise ValueError(
            f"Try to read invalid timestamps {invalid_timestamps} from video {video_path} (FPS: {fps})"
        )
    timestamps = closest_timestamps
    return decoder.get_frames_played_at(seconds=timestamps).data.numpy()


def get_all_frames(
    video_path: str,
    decoder_kwargs: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Get all frames from a video."""
    if not _torchcodec_available():
        import av

        frames_list = []
        pts_list = []
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                frames_list.append(frame.to_ndarray(format="rgb24"))
                pts_list.append(float(frame.time) if frame.time is not None else 0.0)
        return np.stack(frames_list, axis=0), np.asarray(pts_list, dtype=np.float64)

    decoder = _build_decoder(video_path, decoder_kwargs)
    frames = decoder.get_frames_at(indices=range(len(decoder)))
    return frames.data.numpy(), frames.pts_seconds.numpy()

'''

p.write_text(head + new_mid + tail)
for c in p.parent.rglob("__pycache__"):
    import shutil
    shutil.rmtree(c, ignore_errors=True)
print("PATCHED", p)
