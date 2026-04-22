## **Build Instructions**

**1. Overview**



This repository contains a CUDA/C++ implementation for global stiffness matrix assembly of 20-node hexahedral finite elements (Hex20). The code is implemented in a single CUDA source file and can be compiled with the NVIDIA CUDA compiler nvcc.



The implementation relies only on:



CUDA runtime



C++ standard library



NVIDIA GPU with CUDA support



No third-party libraries are required.



**2. Tested environment**



The code is expected to work on a Linux system with an NVIDIA GPU and CUDA Toolkit installed.



Example environment:



OS: Ubuntu 20.04 / 22.04



CUDA Toolkit: 11.x or 12.x



Compiler: nvcc



GPU: Any CUDA-capable NVIDIA GPU



Because performance may depend on the GPU architecture, users are encouraged to compile with an architecture flag matching their device.



**3. Prerequisites**



Before building, please make sure the following components are available.



3.1 NVIDIA GPU driver



An NVIDIA driver must be installed and working properly.



You can verify this by running:



nvidia-smi





If the command works and shows your GPU information, the driver is installed correctly.



3.2 CUDA Toolkit



The CUDA compiler nvcc must be available in your environment.



Check with:



nvcc --version





If this command fails, install the CUDA Toolkit and make sure nvcc is included in your PATH.



3.3 C++ build environment



A standard Linux development environment is recommended. For Ubuntu:



sudo apt update

sudo apt install build-essential





In most cases, nvcc will use the system C++ compiler as its host compiler.



**4. Repository structure**



A minimal repository structure is expected to look like this:



.

├── main.cu

├── README.md

└── LICENSE





If your CUDA source file has a different name, replace main.cu in the commands below with the actual filename.



**5. Compilation**

5.1 Basic compilation



To compile the program with optimization enabled:



nvcc -O3 -std=c++14 main.cu -o fem\_assembly





This command produces an executable named fem\_assembly.



5.2 Compilation with a specific GPU architecture



For better compatibility and performance, compile for the appropriate GPU architecture.



Examples:



For Ampere GPUs

nvcc -O3 -std=c++14 -arch=sm\_80 main.cu -o fem\_assembly



For Turing GPUs

nvcc -O3 -std=c++14 -arch=sm\_75 main.cu -o fem\_assembly



For Volta GPUs

nvcc -O3 -std=c++14 -arch=sm\_70 main.cu -o fem\_assembly





If you are unsure which architecture your GPU uses, you may omit the -arch flag for a first build, or look up the compute capability of your GPU model.



5.3 Optional debug build



For debugging, you may disable aggressive optimization and include debug information:



nvcc -O0 -g -G -std=c++14 main.cu -o fem\_assembly\_debug





Notes:



\-g adds host-side debug symbols



\-G enables device-side debugging



Debug builds are much slower than optimized builds



**6. Running the program**



After successful compilation, run:



./fem\_assembly





If a debug executable was built:



./fem\_assembly\_debug



**7. Recommended build flags**



The following flags are recommended for normal use:



\-O3 -std=c++14





Explanation:



\-O3: enables high optimization for performance



\-std=c++14: enables C++14 support, which is sufficient for this implementation



If needed, users may also add:



\-lineinfo





for profiling:



nvcc -O3 -std=c++14 -lineinfo main.cu -o fem\_assembly





This can help when using profiling tools such as Nsight Systems or Nsight Compute.



**8. Common build issues**

8.1 nvcc: command not found



Cause: CUDA Toolkit is not installed or nvcc is not in PATH.



Fix:



install CUDA Toolkit



ensure the CUDA bin directory is in PATH



For example:



export PATH=/usr/local/cuda/bin:$PATH



8.2 Host compiler version mismatch



Some CUDA versions are compatible only with certain GCC versions.



If nvcc reports a host compiler compatibility error, install a supported GCC version and specify it explicitly:



nvcc -O3 -std=c++14 -ccbin gcc-10 main.cu -o fem\_assembly





Replace gcc-10 with a compiler version supported by your CUDA installation.



8.3 Undefined or unsupported architecture



If the -arch=sm\_xx value is not supported by your installed CUDA version, choose a lower or compatible architecture supported by both your GPU and CUDA Toolkit.



8.4 Runtime CUDA errors



The program already includes CUDA runtime error checking through the CHECK\_CUDA\_ERROR macro. If execution fails, error messages will indicate the source file and line number where the failure occurred.



**9. Build example from scratch**



A complete example workflow is shown below.



Step 1: clone the repository

git clone <your-repository-url>

cd <your-repository-directory>



Step 2: verify the CUDA environment

nvidia-smi

nvcc --version



Step 3: compile

nvcc -O3 -std=c++14 main.cu -o fem\_assembly



Step 4: run

./fem\_assembly



**10. Optional Makefile-based build**



If a Makefile is provided in the repository, users can simply run:



make





and then:



./fem\_assembly





A simple example Makefile is:



TARGET = fem\_assembly

SRC = main.cu

NVCC = nvcc

NVCC\_FLAGS = -O3 -std=c++14



all:

&#x09;$(NVCC) $(NVCC\_FLAGS) $(SRC) -o $(TARGET)



clean:

&#x09;rm -f $(TARGET)



**11. Notes specific to this implementation**



This implementation uses:



single-precision floating point (float)



Hex20 elements with 20 nodes per element



3 degrees of freedom per node



3×3×3 Gaussian integration



CUDA kernels for stiffness assembly and CSR conversion



The code also uses fixed-size internal buffers such as MAX\_LIST\_LENGTH. If the mesh connectivity becomes denser than expected, users may need to increase this constant and recompile.



The kernel launch configuration also includes a manually chosen:



const int BLOCK\_SIZE = 32;





This value was selected conservatively to reduce register pressure for Hex20 element computations. Advanced users may experiment with this parameter for their own hardware.



**12. Reproducibility recommendation**



For reproducible performance results, please report the following when running experiments:



GPU model



driver version



CUDA Toolkit version



nvcc version



operating system



compile command used



mesh size or problem dimensions used in the test



Performance may vary substantially across GPU architectures and CUDA versions.

