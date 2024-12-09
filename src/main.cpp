#include <iostream>
#include <torch/torch.h>

int main() {
    std::cout << "System Detection:\n";
    
    if (torch::cuda::is_available()) {
        std::cout << "GPU available - using CUDA\n";
        auto cuda_device = torch::cuda::current_device();
        std::cout << "Device: " << torch::cuda::get_device_name(cuda_device) << std::endl;
    } 
    #ifdef __APPLE__
    else if (torch::mps::is_available()) {
        std::cout << "GPU available - using Metal (MPS)\n";
    } 
    #endif
    else {
        std::cout << "No GPU detected - using CPU\n";
    }

    // Create a sample tensor - it will automatically use the best available device
    auto tensor = torch::rand({3, 3});
    std::cout << "\nCreated tensor:\n" << tensor << std::endl;

    return 0;
} 