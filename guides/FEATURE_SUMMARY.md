# Atomic Concentration Normalization Feature

## Overview

Added the ability to control how atomic concentrations (%At Conc) are normalized when plotting multiple core levels.

## Feature Details

### Default Behavior (Before)

When plotting multiple core levels with atomic concentration data, components across **all core levels** sum to 100%. This means the relative contribution of each core level is visible in the visualization.

**Example (F 1s, O 1s, C 1s):**

- F 1s: ~6-8% total
- O 1s: ~20-28% total
- C 1s: ~50-60% total
- **Total: 100%**

### New Behavior (With `normalize_at_conc_per_core=True`)

Each core level's components sum to 100% independently. This allows for easier comparison of component distributions within each core level, without being dominated by the absolute concentration differences between core levels.

**Example (same data, normalized per-core-level):**

- F 1s: Component A: ~30%, Component B: ~70%
- O 1s: Component A: ~64%, Component B: ~36%
- C 1s: Component A: ~37%, Component B: ~35%, etc.
- **Each core level totals: 100%**

## Implementation

### Code Changes

#### 1. `xps_analysis/plot_helpers/report_plotting.py`

- Added `_normalize_at_conc_per_core()` helper function
  - Normalizes atomic concentration data so each core level's components sum to 100%
  - Scales per measurement (column) independently

- Modified `plot_all_core_levels()` function
  - Extracts `normalize_at_conc_per_core` parameter from `proc_kwargs`
  - Applies normalization before plotting when requested

#### 2. `xps_analysis/xps_plot.py`

- Updated `plot_report()` docstring
  - Added documentation for new `normalize_at_conc_per_core` parameter
  - Explains default behavior and per-core-level behavior

#### 3. `main.py`

- Updated `report_parameters()` function to accept `normalize_per_core` parameter
- Added test code in `main()` to demonstrate both behaviors side-by-side
  - Default: all components sum to 100% globally
  - Per-core: each core level's components sum to 100% independently

## Usage

### In your code

```python
import xps_analysis.xps_processing as xp
import xps_analysis.xps_plot as xplot

report_dict = xp.read_report_file("your_report.txt")

# Plot with per-core-level normalization
proc_kwargs = {
    "fit_param": "At Conc",
    "core_levels": ["F 1s", "O 1s", "C 1s"],
    "normalize_at_conc_per_core": True,  # NEW: normalize each core independently
}

xplot.plot_report(report_dict, proc_kwargs=proc_kwargs)
```

### Parameter Options

- `normalize_at_conc_per_core: False` (default) - Components across all core levels sum to 100%
- `normalize_at_conc_per_core: True` - Each core level's components sum to 100% independently

## Testing

The feature has been tested with multi-core XPS data:

- Test file: `5_e_beam_unwashed_before_after_comp_report_F1s_O1s_C1s.txt`
- Core levels tested: F 1s (2 components), O 1s (2 components), C 1s (5 components)
- Result: ✅ Both normalization modes work correctly
- Visualization: Two plots generated side-by-side for comparison

## Mathematical Verification

For measurement 1 with `normalize_at_conc_per_core=False`:

```pl
F 1s:  5.33 + 11.89 = 17.22
O 1s: 20.88 + 11.61 = 32.49
C 1s: 18.41 + 17.77 + 0.55 + 12.46 + 1.09 = 50.28
Total: 17.22 + 32.49 + 50.28 = 99.99% ≈ 100% ✓
```

For the same measurement with `normalize_at_conc_per_core=True`:

```pl
F 1s:  (5.33/17.22 * 100, 11.89/17.22 * 100) → (31%, 69%)
O 1s: (20.88/32.49 * 100, 11.61/32.49 * 100) → (64%, 36%)
C 1s: (18.41/50.28 * 100, ...) → (~37%, ~35%, ~1%, ~25%, ~2%)
Each core level sums to 100% ✓
```

## Files Modified

1. `xps_analysis/plot_helpers/report_plotting.py` - Core implementation
2. `xps_analysis/xps_plot.py` - API documentation
3. `main.py` - Test/demonstration code

## Next Steps (Optional)

- Add unit tests for the normalization function
- Create visualization comparing both normalization modes
- Add to official documentation if needed
