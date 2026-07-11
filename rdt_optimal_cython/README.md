# Cython-Accelerated Optimal Restricted Decision Tree

This package provides a **true Cython-compiled** implementation of the Optimal Restricted Decision Tree (RDT) algorithm for maximum performance.

## Features

- **C-level performance** through actual Cython compilation (not just Python)
- **Type-annotated** `.pyx` files with C type declarations
- **Vectorized operations** with NumPy C API
- **GIL-released** critical sections for parallel processing potential
- **Fallback support** - works even if Cython compilation fails (uses pure Python)

## Performance-Critical Functions

The following functions are implemented in Cython with C types:

- `calculate_gini_fast()` - Gini impurity calculation
- `calculate_entropy_fast()` - Entropy calculation  
- `calculate_mse_fast()` - Mean squared error calculation
- `find_best_threshold_vectorized()` - Optimized threshold search
- `split_indices_fast()` - Fast index splitting

## Installation & Build

### Prerequisites

1. **Python 3.7+** with NumPy
2. **Cython** (`pip install cython`)
3. **C Compiler**:
   - **Windows**: Visual Studio Build Tools or Visual Studio (with C++ workload)
   - **Linux**: GCC (`sudo apt-get install build-essential`)
   - **macOS**: Xcode Command Line Tools (`xcode-select --install`)

### Quick Build (Windows)

Simply run the build script:

```bash
cd rdt_optimal_cython
build_cython.bat
```

### Manual Build (All Platforms)

```bash
cd rdt_optimal_cython

# Build in-place (recommended for development)
python setup.py build_ext --inplace

# Or install as a package
pip install -e .
```

### Verify Installation

```python
from rdt_optimal_cython.src import HAS_CYTHON
print(f"Cython compiled: {HAS_CYTHON}")
```

## Usage

```python
from rdt_optimal_cython import OptimalRDTCython
import numpy as np

# Create and train the model
tree = OptimalRDTCython(
    task='classification',
    criterion='gini',
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1
)

# Fit and predict
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)
probabilities = tree.predict_proba(X_test)
```

## Integration with Gradient Boosting

This Cython-optimized tree can be used as a base learner in gradient boosting:

```python
from ensemble.gradient_boosting_rdt_fast import GradientBoostingRDTFast

gb = GradientBoostingRDTFast(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1
)
gb.fit(X_train, y_train)
```

## File Structure

```
rdt_optimal_cython/
├── __init__.py                    # Package initialization
├── optimal_rdt_cython.py          # Main Python class (uses Cython if available)
├── setup.py                       # Cython build configuration
├── build_cython.bat               # Windows build script
├── README.md                      # This file
└── src/
    ├── __init__.py                # Source package initialization
    └── _optimal_tree_cython.pyx   # Cython implementation (C-optimized)
```

## Build Output

After successful compilation, you'll see:

- `src/_optimal_tree_cython.c` - Generated C code
- `src/_optimal_tree_cython.pyd` (Windows) or `src/_optimal_tree_cython.so` (Linux/macOS) - Compiled extension
- `src/_optimal_tree_cython.html` - Annotation file showing Cython/C interaction

## Performance Comparison

Expected speedups over pure Python:

- **Gini calculation**: 3-5x faster
- **Entropy calculation**: 4-6x faster  
- **MSE calculation**: 5-8x faster
- **Overall tree building**: 2-3x faster

## Troubleshooting

### Build Errors

**"Unable to find vcvarsall.bat"** (Windows)
- Install Visual Studio Build Tools: https://visualstudio.microsoft.com/downloads/
- Select "Desktop development with C++" workload

**"Python.h not found"**
- Install Python development headers:
  - Ubuntu: `sudo apt-get install python3-dev`
  - CentOS: `sudo yum install python3-devel`

**"numpy/arrayobject.h not found"**
- Ensure NumPy is installed: `pip install numpy`
- Try upgrading: `pip install --upgrade numpy`

### Import Errors

If the Cython module fails to import, the code will automatically fall back to pure Python implementations (slower but still functional).

To check if Cython is being used:
```python
from rdt_optimal_cython.src import HAS_CYTHON
if HAS_CYTHON:
    print("Using Cython-compiled version ✓")
else:
    print("Using pure Python fallback (slower)")
```

## Development

To modify the Cython code:

1. Edit `src/_optimal_tree_cython.pyx`
2. Rebuild: `python setup.py build_ext --inplace`
3. Test the changes

The build process generates an HTML annotation file showing which lines are pure Python vs C-optimized.

## License

Same as the parent RDT project.

## Citation

If you use this implementation in research, please cite the RDT paper.
