# Filesystem cleanup — September 21, 2026

This stronger pass reclaims **70.55 GB**, additional to the earlier cleanup:

| Action | Space reclaimed |
|---|---:|
| 1,279 archived intermediate checkpoints and completed optimizer states | 6.77 GB |
| 52 clean inactive Git checkouts | 2.60 GB |
| Inactive compiler cache | 0.84 GB |
| Lossless compression of 10,250 retained evidence files/archives | 60.33 GB |

There were 44,770 file deletions. The objective archive's full
hash and every retired member were verified before deletion. All 582 retained
selected-checkpoint and forecast files also match their archive inventory. Each
retired checkout has its commit, retained Git ref and recreation command recorded;
all those commits are in main's history. Recreate historical code paths before
running an old recipe that imports them.

Raw sources and accepted data stores were untouched. Selected models, forecasts
and unique evidence remain. Compression preserves decoded bytes, paths and file
hashes; it adds decompression cost when reading cold historical artifacts. The
active attention/evaluation checkouts and compiler cache remain online.

The first 776 compression checks are reused. One in-flight file completed after
the deliberate speed switch and was independently checked against its existing
hash sidecar. The remaining pass uses faster XPRESS16K compression. This was an
operational change only; no research result or model coordinate changed.

The [receipt](v2_strong_cleanup.json), resolved by `scaling_strong_cleanup`, binds
the per-file journals and a restored/hash-checked metadata ZIP on D:. Existing
complete research archives remain the recovery source for retired checkpoints;
they were not copied into another archive. File-allocation savings exclude
concurrent training growth, so they differ from the change in volume free space.

Training continues separately. Its job-ordering assertion was corrected in the
hash-bound stopping resume, retaining all six completed parents and two completed
children. No recurring heartbeat was enabled.
