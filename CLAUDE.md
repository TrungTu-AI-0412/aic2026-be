# Backend repository instructions

## Product boundary

- This is a competition video retrieval engine, not a chatbot.
- Optimize retrieval accuracy, top-rank quality and query latency.
- Preserve `original_frame_id` throughout the pipeline.
- Qdrant is the vector database and metadata-filtering source.
- No cloud dependency is allowed in the competition query path.

## Architecture rules

- API routes must delegate to application services.
- Qdrant-specific code belongs in `vector_store/`.
- Track modules orchestrate shared retrieval code; do not duplicate it.
- Ingestion must build versioned collections and never modify the active
  collection during competition mode.
- Collection activation requires validation and snapshot completion.
- Parquet manifests remain the rebuild/audit source of truth.

## Review rules

- Keep changes scoped to one implementation-plan item.
- Add tests for mapping, scoring, filtering or API-contract changes.
- Run relevant formatting, linting and tests before handoff.
- Explain new production dependencies before adding them.
- Do not change API contracts silently; regenerate OpenAPI when they change.