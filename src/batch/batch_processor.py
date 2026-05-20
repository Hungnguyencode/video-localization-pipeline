from __future__ import annotations

import csv
import json
import shutil
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


PipelineFactory = Callable[[], Any]
ProgressCallback = Callable[[int, int, "BatchItemResult"], None]


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def make_batch_run_id(prefix: str = "batch") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def safe_name(name: str) -> str:
    keep = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_", ".", " "):
            keep.append(ch)
        else:
            keep.append("_")
    output = "".join(keep).strip().replace(" ", "_")
    return output or "video"


def is_video_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    return value


def _collect_existing_paths(result: Dict[str, Any]) -> Dict[str, str]:
    """
    Lấy các file output mà pipeline đã trả ra.
    Hàm này cố tình rộng một chút để bắt được:
    - localized_video_path
    - bilingual_transcript_path
    - srt_path
    - tts_alignment_report_json_path
    - report_html_path
    ...
    """
    paths: Dict[str, str] = {}

    def visit(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                visit(f"{prefix}.{k}" if prefix else str(k), v)
            return

        if isinstance(obj, (list, tuple)):
            for idx, v in enumerate(obj):
                visit(f"{prefix}[{idx}]", v)
            return

        if not isinstance(obj, (str, Path)):
            return

        text = str(obj)
        key_lower = prefix.lower()
        looks_like_path_key = (
            key_lower.endswith("_path")
            or key_lower.endswith(".path")
            or "path" in key_lower
            or key_lower.endswith("_file")
            or "file" in key_lower
        )
        if not looks_like_path_key:
            return

        p = Path(text)
        if p.exists() and p.is_file():
            paths[prefix] = str(p)

    visit("", result)
    return paths


def _copy_outputs_to_job_dir(result: Dict[str, Any], job_dir: Path) -> Dict[str, str]:
    copied_dir = job_dir / "outputs"
    copied_dir.mkdir(parents=True, exist_ok=True)

    copied: Dict[str, str] = {}
    used_names: set[str] = set()

    for key, src_text in _collect_existing_paths(result).items():
        src = Path(src_text)
        if not src.exists() or not src.is_file():
            continue

        dst_name = safe_name(src.name)
        stem = Path(dst_name).stem
        suffix = Path(dst_name).suffix
        counter = 2
        while dst_name in used_names or (copied_dir / dst_name).exists():
            dst_name = f"{stem}_{counter}{suffix}"
            counter += 1

        dst = copied_dir / dst_name
        shutil.copy2(src, dst)
        used_names.add(dst_name)
        copied[key] = str(dst)

    return copied


@dataclass
class BatchItemResult:
    index: int
    input_path: str
    video_name: str
    status: str = "pending"  # pending | running | success | failed
    started_at: str = ""
    finished_at: str = ""
    elapsed_sec: float = 0.0
    job_dir: str = ""
    copied_outputs: Dict[str, str] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    traceback: str = ""


class BatchProcessor:
    """
    Batch process nhẹ cho demo:
    - nhận nhiều video;
    - xử lý tuần tự bằng pipeline.process_video(video_path);
    - mỗi video có 1 thư mục riêng;
    - cuối batch xuất summary JSON/CSV.

    Lưu ý:
    - Batch này là auto-pipeline, không qua màn sửa thủ công từng segment.
    - Nếu muốn sửa transcript thủ công thì vẫn dùng flow 1 video như hiện tại.
    """

    def __init__(
        self,
        project_root: str | Path,
        pipeline_factory: PipelineFactory,
        output_root: str | Path | None = None,
    ):
        self.project_root = Path(project_root)
        self.pipeline_factory = pipeline_factory
        self.output_root = (
            Path(output_root)
            if output_root is not None
            else self.project_root / "data" / "output" / "batch_runs"
        )
        self.output_root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        video_paths: Iterable[str | Path],
        run_id: str | None = None,
        stop_on_error: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        video_list = [Path(p) for p in video_paths]
        video_list = [p for p in video_list if p.exists() and p.is_file() and is_video_file(p)]

        run_id = run_id or make_batch_run_id()
        run_dir = self.output_root / safe_name(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        results: List[BatchItemResult] = []
        total = len(video_list)

        for idx, video_path in enumerate(video_list, start=1):
            item = BatchItemResult(
                index=idx,
                input_path=str(video_path),
                video_name=video_path.name,
                status="running",
                started_at=datetime.now().isoformat(timespec="seconds"),
                job_dir=str(run_dir / f"{idx:03d}_{safe_name(video_path.stem)}"),
            )
            item_job_dir = Path(item.job_dir)
            item_job_dir.mkdir(parents=True, exist_ok=True)

            t0 = time.time()
            try:
                # Fresh pipeline cho từng video để tránh state/cache UI cũ ảnh hưởng video sau.
                pipeline = self.pipeline_factory()
                raw_result = pipeline.process_video(video_path)

                item.result = _as_jsonable(raw_result or {})
                item.copied_outputs = _copy_outputs_to_job_dir(item.result, item_job_dir)
                item.status = "success"

            except Exception as exc:
                item.status = "failed"
                item.error = f"{type(exc).__name__}: {exc}"
                item.traceback = traceback.format_exc()

                (item_job_dir / "error.txt").write_text(
                    item.traceback,
                    encoding="utf-8",
                )

                if stop_on_error:
                    item.finished_at = datetime.now().isoformat(timespec="seconds")
                    item.elapsed_sec = round(time.time() - t0, 2)
                    results.append(item)
                    if progress_callback:
                        progress_callback(idx, total, item)
                    break

            item.finished_at = datetime.now().isoformat(timespec="seconds")
            item.elapsed_sec = round(time.time() - t0, 2)
            results.append(item)

            # Lưu từng item để đang chạy mà crash vẫn còn log.
            (item_job_dir / "result.json").write_text(
                json.dumps(asdict(item), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            if progress_callback:
                progress_callback(idx, total, item)

        summary = self._write_summary(run_dir, run_id, results)
        return summary

    def _write_summary(
        self,
        run_dir: Path,
        run_id: str,
        results: List[BatchItemResult],
    ) -> Dict[str, Any]:
        items = [asdict(item) for item in results]
        success_count = sum(1 for item in results if item.status == "success")
        failed_count = sum(1 for item in results if item.status == "failed")

        summary = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "total": len(results),
            "success": success_count,
            "failed": failed_count,
            "items": items,
        }

        json_path = run_dir / "batch_summary.json"
        json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        csv_path = run_dir / "batch_summary.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "index",
                    "video_name",
                    "status",
                    "elapsed_sec",
                    "input_path",
                    "job_dir",
                    "error",
                    "copied_outputs_count",
                ],
            )
            writer.writeheader()
            for item in results:
                writer.writerow(
                    {
                        "index": item.index,
                        "video_name": item.video_name,
                        "status": item.status,
                        "elapsed_sec": item.elapsed_sec,
                        "input_path": item.input_path,
                        "job_dir": item.job_dir,
                        "error": item.error,
                        "copied_outputs_count": len(item.copied_outputs),
                    }
                )

        summary["summary_json_path"] = str(json_path)
        summary["summary_csv_path"] = str(csv_path)
        return summary
