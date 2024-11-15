# sundials-sandbox
Template repository for codes that utilize SUNDIALS.

This includes simple implementations of customized N_Vector and SUNLinearSolver modules, along with testing routines for each.  It also includes four SUNDIALS examples, to serve as templates for codes that utilize the ARKODE and CVODE integrators from SUNDIALS:

* N_Vector module: `nvector_serialcomplex.h` and `nvector_serialcomplex.c`, with testing routine `testComplexNVector.c`.

* SUNLinearSolver module: `sunlinsol_sptfqmrcomplex.h` and `sunlinsol_sptfqmrcomplex.c`, with testing routine `testComplexSUNLinearSolver.c`.

This repository provides a simple `CMakeLists.txt` file to compile each of the above codes against a given SUNDIALS installation.

This setup assumes out-of-source builds, and requires that the path for the installed `SUNDIALSConfig.cmake` file be held in the CMake variable  `SUNDIALS_DIR`.  For example, from within this directory:

```
mkdir build
cd build
cmake -DSUNDIALS_DIR=/usr/local/sundials/lib/cmake/sundials ..
make
```

This has been tested using SUNDIALS v7.0.0.  Older SUNDIALS installations may be used by changing to a different branch:

* `main`: designed to work with SUNDIALS v7.0.0 (should work with any v7.x.x)

* `sundials-v6.6.1`: designed to work with SUNDIALS v6.6.1 (should work with any v6.x.x)
