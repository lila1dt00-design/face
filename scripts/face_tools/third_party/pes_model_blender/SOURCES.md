# Legacy PES MODEL parser source

Upstream: https://github.com/the4chancup/pes-model-blender

The retained parser was downloaded on 2026-10-26 from the upstream master branch. This package includes only ModelFile.py, without modifications.

Only ModelFile.py is imported by the bundled legacy adapter. The model geometry, UVs and bone mappings are parsed in Python, without launching Blender or installing a Blender extension.

Local file hashes are recorded in sources.json. Player models and DDS images have separate authors and are attributed in their model metadata; they are not supplied by this parser repository.
