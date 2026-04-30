# Conventions enforcement

This folder contains convention-enforcement utilities.

- `find_global_builders.py`: scans Python files for module-level `build_*`/`make_*`/`create_*` factory functions returning uppercase types, excluding class named constructors.
- `find_type_checking_guards.py`: scans Python files for `if TYPE_CHECKING:` guards.
- `find_any_usage.py`: scans Python files for `Any` usage.
- `find_crlf_line_endings.py`: scans repository files for forbidden CRLF line endings.
- `measure_conventions_metrics.py`: computes an automated metrics snapshot for the measurable subset of `docs/conventions_dods_and_metrics.md` and can emit JSON.

Use:

```bash
python conventions/find_global_builders.py .
python conventions/find_type_checking_guards.py .
python conventions/find_any_usage.py src/answer_engineering
python conventions/find_crlf_line_endings.py .
python conventions/enforcement/measure_conventions_metrics.py \
  --repo-root . \
  --output-json docs/conventions_metrics_snapshot_YYYY-MM-DD.json
```
