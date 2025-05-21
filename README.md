# GPU Kernel programming from scratch using Metal and C++

This project demonstrates how to:
1.  Integrate Metal Performance Shaders (MPS) with C++.
2.  Write and compile a custom Metal GPU kernel, in this case matrix multiplication.

## Compilation

Follow these steps to compile the project:

### 1. Compile the Metal Kernel

First, compile your custom Metal kernel (`matmul.metal` in this example) into an AIR (Assembly Intermediate Representation) file, and then link it into a Metal library (`.metallib`):

```bash
xcrun -sdk macosx metal -c matmul.metal -o matmul.air
xcrun -sdk macosx metallib matmul.air -o default.metallib
```

### 2. Compile the C++/Objective-C++ Code

Next, compile the C++ and Objective-C++ wrapper code, linking against the necessary Metal frameworks and your compiled Metal library:

```bash
clang++ -std=c++17 gpu.cpp matmul_mps.mm -framework Metal -framework Foundation -framework MetalPerformanceShaders -ObjC++ -o mps_matmul_test
```

## Running

After successful compilation, you can run the executable:

```bash
./mps_matmul_test
```

This will execute the MPS-based matrix multiplication and display the results. On my M3 Pro, the I get the following timings for 2048x2048 matrix multiplications:

CPU Time: 21243.9 ms

GPU (unoptimised) Time: 131.83 ms

MPS Optimized GPU Time: 33.05 ms

## Notes

- The `gpu.cpp` file contains the C++ code that integrates with the Objective-C++ wrapper (`matmul_mps.mm`).
- The `matmul.metal` file defines the custom Metal kernel. This is a simple matrix multiplication kernel, there are plenty of ways to optimise this. 
- The `matmul_mps.mm` file is the Objective-C++ wrapper that exposes the kernel to C++ and also the default MPS implementation.
