# Eval records

One JSON file per evaluation run, filed under the bundle it evaluated:
`evals/<bundle>/<embodiment>-<setting>-<date>.json`. The format is in
[SPEC.md](../SPEC.md#eval-records). Records set a bundle's maturity:

- `community`: no passed record yet; every bundle starts here.
- `sim-verified`: a passed record from simulation (`setting: sim`), from a
  RoboSimStudio suite run under the stated tier.
- `real-verified`: a passed record from a real cell (`setting: real`), with
  evidence (a trace, a video) linked.

Only maintainers merge records. The plan is for CI to write `sim` records
itself, from GPU runners that run each bundle's suite on every release.
