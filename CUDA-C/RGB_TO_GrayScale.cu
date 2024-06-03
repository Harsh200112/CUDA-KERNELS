#include<iostream>
#include<cuda_runtime.h>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

__global__ void rgb_to_grayscale_kernel(unsigned char *image, float *output, int width, int height){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= 0 && x < width && y >= 0 && y < height){
        int i = y * width + x;
        output[i] = image[3 * i] * 0.21 + image[3 * i + 1] * 0.63 + image[3 * i + 2] * 0.16;
    }
}

void rgb_to_grayscale(unsigned char *image, float *output, int width, int height){
    unsigned char *image_d;
    float *output_d;
    size_t size = width * height * sizeof(unsigned char);
    
    cudaMalloc((void**)&image_d, size * 3);
    cudaMalloc((void**)&output_d, size);

    cudaMemcpy(image_d, image, size * 3, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockIdx.x -1)/ blockIdx.x, (height + blockIdx.y -1) / blockIdx.y);

    rgb_to_grayscale_kernel<<<gridSize, blockSize>>>(image_d, output_d, width, height);

    cudaMemcpy(output, output_d, size, cudaMemcpyDeviceToHost);

    cudaFree(image_d);
    cudaFree(output_d);
}

void displayImage(windowName, image){
    namedWindow(windowName);
    imshow(windowName, image);
    waitKey(0);
    destroyWindow(windowName);
}

int main(){
    Mat image = imread("C:/Users/Harsh Soni/Downloads/CUDA Kernel Project/test.png")

    if (image.empty()){
        cout<<"Could not find the image."<<endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    int size = width * height;

    unsigned char *image_data = image.data;
    float *output = new float[size];

    rgb_to_grayscale(image_data, output, width, height);

    Mat output_image;
    reshape(output, output_image, width, height);
    String windowName = "Gray Image";
    
    display(windowName, output_image);

    return 0;
}