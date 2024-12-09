#include <iostream>
#include <string>

#ifdef _WIN32
    #include <windows.h>
    #ifdef USE_CUDA
        #include <cuda_runtime.h>
    #endif
#endif

#ifdef __APPLE__
    #include <sys/sysctl.h>
#endif

// Function to check if running on Apple Silicon
bool isAppleSilicon() {
#ifdef __APPLE__
    char buffer[100];
    size_t size = sizeof(buffer);
    if (sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nullptr, 0) == 0) {
        return std::string(buffer).find("Apple") != std::string::npos;
    }
#endif
    return false;
}

// Function to check for CUDA GPU
bool hasNvidiaGPU() {
#ifdef _WIN32
    #ifdef USE_CUDA
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        return (error == cudaSuccess && deviceCount > 0);
    #endif
#endif
    return false;
}

bool selectGPU(int deviceIndex = 0) {
#ifdef _WIN32
    #ifdef USE_CUDA
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error == cudaSuccess && deviceIndex < deviceCount) {
            cudaSetDevice(deviceIndex);
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, deviceIndex);
            std::cout << "Using GPU: " << prop.name << "\n";
            return true;
        }
    #endif
#endif
    return false;
}

int main() {
    std::cout << "System Detection:\n";
    
    #ifdef __APPLE__
        std::cout << "Running on macOS\n";
        if (isAppleSilicon()) {
            std::cout << "Apple Silicon detected\n";
            // Initialize Metal or other Apple Silicon specific code here
        } else {
            std::cout << "Running on Intel Mac\n";
        }
    #endif

    #ifdef _WIN32
        std::cout << "Running on Windows\n";
        if (hasNvidiaGPU()) {
            std::cout << "NVIDIA GPU with CUDA support detected\n";
            // Initialize CUDA specific code here
        } else {
            std::cout << "No NVIDIA GPU detected or CUDA not available\n";
        }
    #endif

    // Your common application code here
    
    return 0;
} 