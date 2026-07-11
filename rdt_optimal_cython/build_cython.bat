@echo off
REM Build script for Cython extensions on Windows

echo ========================================
echo Building Cython Extensions for RDT
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "setup.py" (
    echo ERROR: setup.py not found!
    echo Please run this script from the rdt_optimal_cython directory
    pause
    exit /b 1
)

echo Step 1: Cleaning previous builds...
if exist "build" rmdir /s /q build
if exist "src\_optimal_tree_cython.c" del /q src\_optimal_tree_cython.c
if exist "src\_optimal_tree_cython.pyd" del /q src\_optimal_tree_cython.pyd
if exist "src\_optimal_tree_cython*.so" del /q src\_optimal_tree_cython*.so
echo Done.
echo.

echo Step 2: Building Cython extensions...
python setup.py build_ext --inplace
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Build failed!
    echo.
    echo Common issues:
    echo - Missing C compiler: Install Visual Studio Build Tools
    echo - Missing Cython: pip install cython
    echo - Missing NumPy: pip install numpy
    pause
    exit /b 1
)
echo Done.
echo.

echo Step 3: Verifying build...
python -c "from src._optimal_tree_cython import calculate_gini_fast; print('✓ Cython module imported successfully!')"
if %ERRORLEVEL% neq 0 (
    echo.
    echo WARNING: Import test failed!
    pause
    exit /b 1
)
echo.

echo ========================================
echo ✓ BUILD SUCCESSFUL!
echo ========================================
echo.
echo The Cython extensions are now compiled and ready to use.
echo You can now use the OptimalRDTCython class with C-level performance.
echo.
pause
