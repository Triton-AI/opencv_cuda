import cv2
import numpy as np
import time

def cuda_stress_test(duration_sec=30):
    height, width, channels = 1080, 1920, 3
    gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (15, 15), 0)
    start_time = time.time()
    iterations = 0
    while time.time() - start_time < duration_sec:
        img = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        gpu_result = gaussian_filter.apply(gpu_img)
        result = gpu_result.download()
        iterations += 1
        if iterations % 10 == 0:
            print(f"Iteration {iterations} completed")
    print(f"Completed {iterations} iterations in {duration_sec} seconds.")

if __name__ == "__main__":
    print("Starting CUDA stress test...")
    cuda_stress_test(duration_sec=30)
    print("Stress test complete.")
