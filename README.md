# Option Pricing Using Fast Fourier Transform (FFT)

An interactive quantitative finance dashboard that implements the landmark Carr-Madan (1999) methodology to price European options under the Variance Gamma (VG) stochastic process, benchmarked against the analytical Black-Scholes-Merton model.

---

## 📌 Project Overview
This repository contains a full computational implementation developed as an academic exposition for the course **DMS613(Introduction to Mathematical Finance)** under the supervision of **Prof. Sourav Majumdar**. 

The main objective of this project was to explore how shifting option pricing problems from the time domain into the frequency domain allows for the simultaneous calculation of an entire chain of option strikes with remarkable speed ($O(N \log N)$ complexity).

### Key Features Implemented:
* **Variance Gamma Pricing Engine:** Models asset returns using a jump-diffusion process that captures real-world skewness ($\theta$) and kurtosis ($\nu$), neutralizing the constant-volatility bias of Black-Scholes.
* **Carr-Madan Damping:** Implements an exponential damping factor ($\alpha$) to handle the non-square integrability of the raw call payoff function.
* **Hyperbolic Smoothing:** Integrates a modified time-value isolation technique using $\sinh$ damping to eliminate high-frequency pricing oscillations found in short-maturity options ($T \to 0$).
* **Numerical Integration:** Discretizes the continuous pricing integral using Simpson's Rule, matching the Nyquist grid restriction ($\lambda \eta = 2\pi/N$) required by the Discrete Fourier Transform.
* **Streamlit Dashboard:** A visual interface to dynamically adjust market inputs, tweak model parameters, and directly observe the "volatility smile" discrepancies against the Black-Scholes baseline.

---

## 📌 Mathematical Architecture

Traditional option pricing requires integrating a complex Probability Density Function (PDF). Under advanced models like Variance Gamma, the PDF is non-analytical, but its **Characteristic Function** ($\phi_T(u)$) is simple and explicit.

### 1. The Damped Integrand
Because an option's payoff does not vanish as the log-strike $k \to -\infty$, it is not square-integrable. By pricing a modified call $c_T(k) \equiv e^{\alpha k}C_T(k)$, we derive the Fourier transform $\psi_T(v)$:

$$\psi_T(v) = \frac{e^{-rT}\phi_T(v - (\alpha+1)i)}{\alpha^2 + \alpha - v^2 + i(2\alpha+1)v}$$

### 2. Discretization for FFT
Using the Trapezoid/Simpson's rule with an integration step size $\eta$ and a log-strike grid spacing of $\lambda$, we map the continuous inverse transform onto a discrete array structure that perfectly mirrors a Discrete Fourier Transform (DFT) summation layout:

$$C_T(k_u) \approx \frac{e^{-\alpha k_u}}{\pi}\sum_{j=1}^{N}e^{-i\frac{2\pi}{N}(j-1)(u-1)}e^{ibv_j}\psi_T(v_j)\frac{\eta}{3}\left[3+(-1)^j-\delta_{j-1}\right]$$

To achieve valid structural alignment for the fast algorithm, the grid parameters must satisfy the constraint:
$$\lambda\eta = \frac{2\pi}{N}$$

---

## 📌 Tech Stack & Dependencies

* **Language:** Python 3.8+
* **Core Numerics:** NumPy, SciPy (specifically `scipy.fft.fft` and `scipy.stats.norm`)
* **Data Processing:** Pandas
* **UI/Visualization:** Streamlit

To install dependencies, run:
```bash
pip install streamlit numpy pandas scipy
```

## 📌 How to Run the Web Application

1. Clone the repository to your local machine:
   ```bash
   git clone [https://github.com/Shikhardg24/Option-Pricing-FFT.git](https://github.com/Shikhardg24/Option-Pricing-FFT.git)
   cd Option-Pricing-Using-Fast-Fourier-Transform-FFT-
   ```
2. Launch the local Streamlit server:
   ```bash
   streamlit run Pricing_Engine.py
   ```
3. Open your browser to the local URL (usually `http://localhost:8501`) to interact with the parameters.
## 📌 Sample Insights & Experimentation

By adjusting the **Kurtosis ($\nu$)** and **Skewness ($\theta$)** sliders in the sidebar, you can instantly see how the Variance Gamma model handles non-normal asset return profiles:
* Setting $\nu \to 0$ and $\theta \to 0$ aligns the pricing engine back toward log-normal behaviors.
* Adjusting $\theta < 0$ tilts the pricing curve, visually highlighting the absolute discrepancies where the Black-Scholes benchmark systematically underprices out-of-the-money options due to real-world volatility distortions.

---
