# Redacted Usage Artifacts

This directory contains redacted aggregate billing and usage exports used for
the thesis cost-accounting appendix.

Redaction policy:

- Provider project, organization, user, email, and API-key identifiers are
  removed.
- Per-user and per-API-key breakdowns are removed.
- Aggregate dates, service/model labels, request counts, token counts, costs,
  and currency-conversion metadata are retained.
- Raw unredacted exports should be kept outside the repository or under
  `usage/raw/`, which is ignored by Git.

These files are suitable for archival release as aggregate reproducibility
metadata. They are not prompt-, model-, or language-level cost measurements.
