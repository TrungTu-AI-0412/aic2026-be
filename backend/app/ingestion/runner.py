import argparse
import sys

from app.core.config import settings
from app.ingestion import pipeline, store
from app.schemas.ingestions import IngestionEntity


def run(job_id: str, db_path: str) -> None:
    job = store.get_job(db_path, job_id)
    if job is None:
        print(f"job {job_id} not found", file=sys.stderr)
        sys.exit(1)

    entity = IngestionEntity(job["entity"])

    try:
        store.update_job(db_path, job_id, status="running", stage="validating")
        total = pipeline.validate_manifest(job["manifest_path"], entity)

        store.update_job(db_path, job_id, stage="creating_collection")
        pipeline.create_collection(
            job["collection_name"], job["feature_profile"], entity
        )

        store.update_job(db_path, job_id, stage="creating_payload_indexes")
        pipeline.create_payload_indexes(job["collection_name"], entity)

        store.update_job(
            db_path,
            job_id,
            stage="upserting",
            progress_completed=0,
            progress_total=total,
        )

        def on_progress(completed: int) -> None:
            store.update_job(db_path, job_id, progress_completed=completed)

        pipeline.upsert_points(
            job["collection_name"],
            job["manifest_path"],
            entity,
            job["feature_profile"],
            on_progress,
        )

        store.update_job(db_path, job_id, stage="optimizing")
        pipeline.optimize_collection(job["collection_name"])

        store.update_job(db_path, job_id, status="succeeded", stage="completed")
    except Exception as exc:
        store.update_job(db_path, job_id, status="failed", error=str(exc))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--db-path", default=settings.INGESTION_DB_PATH)
    args = parser.parse_args()

    run(args.job_id, args.db_path)


if __name__ == "__main__":
    main()
