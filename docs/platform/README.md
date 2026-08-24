# SemiRestore platform handoff

The platform branch provides a secure, observable FastAPI service shell around a
narrow model-service protocol. It does not contain the final
`SemiRestorePipeline`, load a real checkpoint, or claim scientific restoration
correctness. Without a real adapter, the application deliberately runs live but
unready and rejects restoration work.

## Documentation index

The focused documents below are authoritative for their subject. This index maps
the final handoff requirements without repeating their details.

1. [Architecture](architecture.md#architecture)
2. [API contract](api-contract.md#endpoints)
3. [Runtime settings](operations.md#runtime-settings)
4. [Local development](operations.md#local-development)
5. [Application startup and shutdown](architecture.md#application-lifecycle)
6. [Health and readiness semantics](api-contract.md#health-and-readiness)
7. [Error contract](api-contract.md#error-contract)
8. [Upload security limits](api-contract.md#upload-security)
9. [Restoration request and response](api-contract.md#restoration-contract)
10. [Base64 response trade-off](api-contract.md#base64-transport)
11. [Model-service adapter boundary](model-adapter-contract.md)
12. [Missing checkpoint behavior](model-adapter-contract.md#missing-or-invalid-checkpoints)
13. [Request IDs](operations.md#request-ids)
14. [Structured logging](operations.md#structured-logging)
15. [Metrics](operations.md#metrics)
16. [Concurrency and backpressure](operations.md#concurrency-and-backpressure)
17. [Timeout behavior](operations.md#timeouts-and-cancellation)
18. [CPU deployment](deployment.md#cpu-deployment)
19. [GPU limitations](deployment.md#gpu-limitations)
20. [Docker usage](deployment.md#docker-usage)
    - [Local Compose stack](local-stack.md)
21. [Docker health checking](deployment.md#container-health-check)
22. [Runtime checkpoint mounting](deployment.md#runtime-model-artifacts)
23. [Privacy and non-persistence](architecture.md#privacy-boundary)
24. [Known limitations](deployment.md#known-limitations)
25. [Teammate integration checklist](integration-checklist.md)

## Ownership boundary

Platform-owned code lives in `src/semirestore/platform/` and
`src/semirestore/api/`, with tests in `tests/platform/` and `tests/api/`.
Model architecture, checkpoint contents and compatibility, scientific image
processing, inference behavior, training, evaluation, diagnostic meaning, and
restoration-quality claims remain model-owned.

The immediate integration target is the contract in
[model-adapter-contract.md](model-adapter-contract.md). The final checklist lists
the exact work that remains before a real deployment can become ready.
