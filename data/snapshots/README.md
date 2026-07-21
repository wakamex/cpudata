# CPU observation snapshots

Each timestamped JSON file is an immutable capture of the PassMark desktop CPU
table before analysis filters are applied. Snapshots include rows without prices,
the source-content hash, units, and the Git/GitHub Actions run that captured them.

Published results in `docs/data.json` reference their input snapshot by path and
SHA-256. Prices are stored as integer USD cents so historical fits can be rebuilt
without floating-point parsing ambiguity.
