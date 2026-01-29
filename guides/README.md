# XPS Analysis Package Documentation

This directory contains additional documentation and guides for the `battery-xps-analysis` package.

## Available Documentation

### FEATURE_SUMMARY.md

Documents the atomic concentration normalization feature that allows you to control how %At Conc values are normalized when plotting multiple core levels:

- **Default behavior**: Components across all core levels sum to 100%
- **Per-core normalization**: Each core level's components sum to 100% independently

This is useful when comparing component distributions within each core level without being dominated by absolute concentration differences between core levels.

**Key parameter**: `normalize_at_conc_per_core` in `proc_kwargs`

## Additional Resources

- **Main README**: See `../README.md` for installation, quick start, and general usage
- **API Documentation**: Module docstrings in the source code provide detailed API reference
- **Example Script**: See `../main.py` for example usage and workflow patterns

## Package Architecture

The package follows a modular architecture with clear separation of concerns:

- **Processing logic** (`processing_helpers/`): File parsing, table processing, data transformation
- **Plotting logic** (`plot_helpers/`): Visualization, axis management, figure rendering
- **Public API** (`xps_processing.py`, `xps_plot.py`, `xps_utilities.py`): High-level user-facing functions

## Contributing Documentation

When adding new features to the package:

1. Update module docstrings with new functions/parameters
2. Add examples to `main.py` if applicable
3. Create or update documentation in this `guides/` folder for significant features
4. Update the main `README.md` if the feature affects the quick start or main workflow
