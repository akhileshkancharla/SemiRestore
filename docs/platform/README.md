# SemiRestore platform handoff

The platform provides a secure, observable FastAPI service around a narrow
model-service protocol and the integrated production `SemiRestorePipeline`
adapter. The verified checkpoint remains an ignored runtime dependency. When it
is absent or invalid, the application deliberately stays live but unready and
rejects restoration work without activating a fake. Neither the platform nor
the dashboard claims scientific restoration correctness: restored images are
not ground truth, and suitability guidance is advisory. Uploads and restored
outputs are processed in memory and are not permanently stored by default.

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
18. [Checkpoint installation](deployment.md#checkpoint-installation)
19. [Local CPU startup](deployment.md#local-cpu-startup)
20. [CPU deployment](deployment.md#cpu-deployment)
21. [GPU limitations](deployment.md#gpu-limitations)
22. [Docker usage](deployment.md#docker-usage)
    - [Local Compose stack](local-stack.md)
23. [Docker health checking](deployment.md#container-health-check)
24. [Runtime checkpoint mounting](deployment.md#runtime-model-artifacts)
25. [Privacy and non-persistence](architecture.md#privacy-boundary)
26. [Known limitations](deployment.md#known-limitations)
27. [Teammate integration checklist](integration-checklist.md)
28. [Dashboard and local Compose stack](local-stack.md#dashboard-usage)
29. [Continuous integration](ci.md)
30. [Load and resilience testing](load-testing.md)
31. [End-to-end smoke testing](smoke-testing.md)
32. [Environment-variable reference](environment.md)
33. [Operations runbook](runbook.md)
34. [Troubleshooting](troubleshooting.md)

## Ownership boundary

Platform-owned code lives in `src/semirestore/platform/` and
`src/semirestore/api/`, with tests in `tests/platform/` and `tests/api/`.
Model architecture, checkpoint contents and compatibility, scientific image
processing, inference behavior, training, evaluation, diagnostic meaning, and
restoration-quality claims remain model-owned. The platform adapter imports
those reviewed model APIs but does not reimplement them.

The implemented integration is described in
[model-adapter-contract.md](model-adapter-contract.md). The final checklist lists
the remaining environment- and hardware-dependent release gates.
