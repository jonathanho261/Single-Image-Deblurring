# Single-Image-Deblurring-with-Adaptive-Dictionary-Learning

#### Organization of Project
* Deblur.ipynb: Jupyter Notebook that runs the main learning algorithm
    - direct_sparse(img, D, kernel, num_patches, skip)
        - Runs the direct sparse algoritm by Yifei Lou et al. that solves the deblur problem with known kernel. This function also tries to reconstruct the image from a weighted sum of patched.
    - estimate_kernel(blur_img, I, kernel_size, lambda_=5)
        - Estimates the blur kernel by solving with Tikhonov regularization
    - deblur(blur_img, kernel_size, patch_size, iters, blur_strength, skip)
        - Deblur a single image with adapative dictionary learning. Algorithm described by Hu et al.

* utils/utils.py: Python file that contains helper functions not related to algorithm
    - gaussian_kernel(sigma, kernel_half_size)
        - Returns a 2D Gaussian kernel matrix
    - get_patch_indicies(image, i, j, patch_size)
        - Returns a patch of size patch_size that is within the bounds of the image

* utils/odctdict.py: Python file that creates an overcomplete DCT dictionary
    - odctdict(N, L)
        - Creates an overcomplete DCT dictionary 


#### How to run code and recreate my results:
```python
kernel_size   = 7
patch_size    = 7
iters         = 6
blur_strength = 0.4
skip          = (3,3)

blur_img = cv2.cvtColor(cv2.imread('data/blurred.jpg'), cv2.COLOR_BGRA2GRAY).astype('double') / 255.0

deblur_img, kernel = deblur(blur_img, kernel_size=kernel_size, patch_size=patch_size, iters=iters, blur_strength=blur_strength, skip=skip)

output_img = restoration.richardson_lucy(blur_img, kernel)
plt.imshow(output_img, cmap='gray')
plt.show()
```

#### To install all packages used in this project:
```
pip install -r requirements.txt
```

#### Project Report and Acknowledgements
https://docs.google.com/document/d/14u6S1osvJHnJnakIf7TdsWBOcf2fvOYLOCneCbLOC5M/edit?usp=sharing