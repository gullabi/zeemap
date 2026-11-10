# Examples

- `synth_single.toml`: a one-phase fit of the synthetic toy star made by `tools/zeeman_library.py`. The comments at
  the top of the file give the commands; the README's quick start walks through it.

Configurations use relative paths only: `library.path` and `data.path` are relative to `$ZEEMAP_DATA`, the phase files
to `data.path`. Every key not set takes its default from `src/zeemap/defaults.toml`. Keep configurations for real
stars next to their data, outside the repository.
