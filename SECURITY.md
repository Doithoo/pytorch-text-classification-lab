# Security Policy

## Supported version

Security fixes are applied to the latest revision on `main`. The project is an educational lab and does not currently publish long-term-supported release branches.

## Reporting a vulnerability

Do not disclose a vulnerability in a public issue before maintainers can assess it. Use GitHub's private vulnerability reporting for `Doithoo/pytorch-text-classification-lab` when available. Include affected revision, environment, reproduction steps, impact, and a minimal proof of concept without private data.

## Trust boundaries

PyTorch `.pt` training checkpoints use pickle and may execute code while loading. The CLI deliberately describes `.pt` inputs as trusted. Never load an unknown user's training checkpoint or expose `evaluate`, `export-inference`, or `resume` as an unauthenticated upload service.

For inference distribution, use `export-inference` to create `.safetensors` weights and a JSON sidecar. This avoids executable pickle during weight loading, but provenance and integrity still matter. The sidecar contains the complete vocabulary and may be a derived data asset.

The download script verifies fixed SHA-256 values, but network and upstream repository trust still matter. Kaggle credentials and API tokens must stay outside the repository. Error-analysis files may contain complete source text; review privacy and dataset terms before sharing them.

This project is not a hardened serving system. It does not provide sandboxing, authentication, rate limiting, adversarial-input defenses, or calibrated safety decisions.
