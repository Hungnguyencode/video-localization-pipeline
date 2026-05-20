from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main_pipeline import VideoLocalizationPipeline
from src.batch.batch_processor import BatchProcessor, make_batch_run_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch process multiple videos sequentially."
    )
    parser.add_argument("videos", nargs="+", help="Video paths")
    parser.add_argument("--run-id", default=make_batch_run_id(), help="Batch run id/name")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop when one video fails")
    args = parser.parse_args()

    processor = BatchProcessor(
        project_root=PROJECT_ROOT,
        pipeline_factory=lambda: VideoLocalizationPipeline(),
    )

    summary = processor.run(
        args.videos,
        run_id=args.run_id,
        stop_on_error=args.stop_on_error,
    )

    print("=== Batch finished ===")
    print(f"Run dir: {summary['run_dir']}")
    print(f"Success: {summary['success']}")
    print(f"Failed: {summary['failed']}")
    print(f"Summary JSON: {summary['summary_json_path']}")
    print(f"Summary CSV: {summary['summary_csv_path']}")


if __name__ == "__main__":
    main()