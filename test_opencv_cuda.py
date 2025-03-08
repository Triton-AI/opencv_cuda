import cv2
import numpy as np

def main():
    print("OpenCV version:", cv2.__version__)
    cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
    if cuda_device_count > 0:
        print("Number of CUDA enabled devices:", cuda_device_count)
        cv2.cuda.setDevice(0)
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        print("Created random image with shape:", img.shape)
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        gauss_filter = cv2.cuda.createGaussianFilter(gpu_img.type(), -1, (5, 5), 0)
        gpu_result = gauss_filter.apply(gpu_img)
        result = gpu_result.download()
        print("CUDA test passed. Processed image shape:", result.shape)
    else:
        print("CUDA is not available in this OpenCV build.")

if __name__ == "__main__":
    main()
