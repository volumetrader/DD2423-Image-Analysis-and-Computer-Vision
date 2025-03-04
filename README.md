
# DD2324 Image analysis

Repository of completed assignments in the course *DD2423 Image Processing and Computer Vision*

## Laboration 1 - Filtering operations

### Fourier transforms
<p>
    <img src="images/ft.jpg" alt="drawing" width="350" />
    <br><em>Continuous fourier transform</em>
</p>

<p>
    <img src="images/discrete_ft.jpg" alt="drawing" width="350" />
    <br><em>Discrete fourier transform</em>
</p>

### Common Properties of Fourier Transforms

>#### 1. Linearity
>The Fourier transform is a linear operation:
>**F{ a f(x) + b g(x) } = a F(ω) + b G(ω)**
>
>#### 2. Time Shifting
>Shifting a function in time corresponds to a phase shift in the Fourier domain:
>**F{ f(x - x₀) } = e^(-iωx₀) F(ω)**
>
>#### 3. Frequency Shifting
>Multiplying by a complex exponential shifts the frequency domain:
>**F{ e^(iω₀x) f(x) } = F(ω - ω₀)**
>
>#### 4. Scaling
>Scaling in time scales frequency inversely:
>**F{ f(ax) } = (1 / |a|) F(ω / a)**
>
>#### 5. Conjugation
>For the complex conjugate of a function:
>**F{ f*(x) } = F*(-ω)**
>
>#### 6. Differentiation Property
>Differentiation in the time domain corresponds to multiplication in the frequency domain:
>**F{ d/dx f(x) } = iω F(ω)**
>
>#### 7. Convolution Theorem
>The Fourier transform of a convolution is the product of Fourier transforms:
>**F{ f(x) * g(x) } = F(ω) ⋅ G(ω)**
>
>#### 8. Multiplication (Modulation) Theorem
>Multiplication in the time domain corresponds to convolution in the frequency domain:
>**F{ f(x) ⋅ g(x) } = F(ω) * G(ω)**

### Sap noise filtering
<img src="images/noise/sap_noise1.jpg" alt="drawing" width="200"/>
<!---<img src="images/noise/sap_noise2.jpg" alt="drawing" width="200"/>
-->
<img src="images/noise/sap_noise3.jpg" alt="drawing" width="200"/>

**Left**: image with salt and pepper noise<br>
**Right**: Median filter applied to noisy image

### Effect of Gauss-smoothing
<img src="images/noise/gauss.jpg" alt="drawing" width="600"/>

**Top**: Down-sampling <br>
**Bottom**: Down-sampling with gauss-smoothing

## Laboration 2 - Edge detection & Hough transform

### Difference operators
<img src="images/edges/diff1.jpg" alt="drawing" height="300"/>
<img src="images/edges/diff2.jpg" alt="drawing" height="300"/>

Output of a difference operator with different minimum thresholds.

### Edge detection
<img src="images/edges/edge_detection.jpg" alt="drawing" width="400"/>

Edge detction combining difference and higher order derivatives

### Hough transform
<img src="images/edges/hough_lines.jpg" alt="drawing" width="250"/>

Computing hough lines in direction of most edges

## Laboration 3 -Image segmentation

### K-Means
<img src="images/segmentation/kmeans_orange1.jpg" alt="drawing" width="250"/>

### Mean-shift segmentation
<img src="images/segmentation/mean_shift.png" alt="drawing" width="250"/>
<img src="images/segmentation/mean_shift2.png" alt="drawing" width="250"/>

### Normalized cut
<img src="images/segmentation/normcuts2.png" alt="drawing" width="250"/>
<img src="images/segmentation/normcuts1.png" alt="drawing" width="250"/>

### Graph cut
<img src="images/segmentation/graphcut.png" alt="drawing" width="250"/>
<img src="images/segmentation/graphcut2.png" alt="drawing" width="250"/>
