"""Wrapper module to expose JSON export functionality.

This file simply re-exports the `export_simulation` function from the
`exporters.json_export` module so that other parts of the codebase can import
`export_simulation` via `from json_exporter import export_simulation`.
"""

from exporters.json_export import export_simulation
