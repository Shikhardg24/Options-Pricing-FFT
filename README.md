# Option Pricing Using Fast Fourier Transform (FFT)

An interactive quantitative finance dashboard that implements the landmark Carr-Madan (1999) methodology to price European **calls and puts** under the Variance Gamma (VG) stochastic process, benchmarked against the analytical Black-Scholes-Merton model.

---

## 📌 Project Overview

This repository contains a full computational implementation developed as an academic exposition for the course **DMS613 (Introduction to Mathematical Finance)** under the supervision of **Prof. Sourav Majumdar**.

The main objective of this project was to explore how shifting option pricing problems from the time domain into the frequency domain allows for the simultaneous calculation of an entire chain of option strikes with remarkable speed ($O(N \log N)$ complexity), and to compare the resulting prices against the constant-volatility Black-Scholes benchmark.

### Key Features Implemented:
* **Variance Gamma Pricing Engine:** Models the log-asset price as a Brownian motion evaluated at a random, Gamma-distributed "business time," which produces skewness ($\theta$) and excess kurtosis ($\nu$) that Black-Scholes' constant-volatility, log-normal assumption cannot capture.
* **Carr-Madan Damping:** Implements an exponential damping factor ($\alpha$) to handle the non-square integrability of the raw call payoff function.
* **Put Pricing via Put-Call Parity:** Derives put prices directly from the FFT-computed call prices using the model-independent no-arbitrage identity $\text{Call} - \text{Put} = S_0e^{-qT} - Ke^{-rT}$, avoiding a second Fourier pass. A UI toggle switches the whole dashboard between Call and Put view.
* **Numerical Integration:** Discretizes the continuous pricing integral using Simpson's Rule, matching the grid restriction ($\lambda \eta = 2\pi/N$) required by the Discrete Fourier Transform.
* **Implied Volatility Smile:** Inverts the Black-Scholes formula against the VG-generated prices at every strike to plot the implied volatility smile the model produces, directly against Black-Scholes' flat input volatility.
* **Distributional Diagnostics:** Displays the exact annualized skewness and excess kurtosis implied by the chosen VG parameters, computed analytically from the process's cumulants.
* **Streamlit Dashboard:** A visual interface to dynamically adjust market inputs and model parameters, toggle between calls and puts, control the displayed strike range, and export results to CSV.

---

## 📌 What's New in This Version

This version is a correctness and feature revision of the original implementation, produced after a full line-by-line audit of the pricing math against known analytical benchmarks. Summary of changes:

**Bug fixes:**
* **Fixed a significant pricing bug** in the call-price finalization step. The original code treated the raw Carr-Madan FFT output for in-the-money strikes as an out-of-the-money *put* price and applied a put-call parity conversion on top of it. Since the damped ($\alpha > 0$) transform already returns the correct *call* price at every strike, this was double-counting intrinsic value — inflating in-the-money call prices by roughly 2x. Verified via a diffusion-limit convergence test (VG with $\theta, \nu \to 0$ must converge to Black-Scholes): pre-fix, in-the-money strikes were off by ~2x; post-fix, they match Black-Scholes to within ~$0.0002.
* **Fixed an incorrect Simpson's-rule endpoint weight.** The last frequency-grid node's quadrature weight was being force-set to $1/3$; per the standard composite Simpson's rule used in the Carr-Madan discretization, only the *first* node receives that correction. Removed the erroneous override.
* Minor: fixed invalid escape-sequence warnings in two LaTeX-labeled slider strings.

**New features:**
* Put pricing via put-call parity, with a Call/Put toggle in the UI (see Key Features above).
* Implied volatility smile chart (VG-implied vs. flat Black-Scholes input).
* Live skewness / excess kurtosis metrics derived from the VG parameters.
* `@st.cache_data` on the FFT pricing pipeline, so unrelated UI interactions (e.g. adjusting the display window) don't silently re-run the full transform.
* User-controlled strike display window (previously a hardcoded ±30% band).
* CSV export of the displayed comparison table.
* An in-app "How this works" explainer summarizing the damping factor, Simpson's rule, and the FFT speedup.

A full, from-first-principles write-up of the theory, the code, the bugs, and the fixes — written to be understandable with no prior options background — is included as `Project_Documentation.md` in this repository.

> **Note on the previous README:** an earlier version of this document described a "jump-diffusion process" with a "hyperbolic smoothing ($\sinh$ damping)" step for short-maturity oscillation control. Variance Gamma is more precisely a **pure-jump Lévy process** (infinite activity, no continuous diffusion component) rather than a jump-diffusion model, and the $\sinh$-damped short-maturity formulation is not part of the current implementation — the engine uses the standard $\alpha$-damped Carr-Madan transform described below at all maturities. Both points are corrected here for accuracy.

---

## 📌 Mathematical Architecture

Traditional option pricing requires integrating a complex Probability Density Function (PDF). Under advanced models like Variance Gamma, the PDF is non-analytical, but its **Characteristic Function** ($\phi_T(u)$) is simple and explicit.

### 1. The Damped Integrand
Because an option's payoff does not vanish as the log-strike $k \to -\infty$, it is not square-integrable. By pricing a modified call $c_T(k) \equiv e^{\alpha k}C_T(k)$, we derive the Fourier transform $\psi_T(v)$:

$$\psi_T(v) = \frac{e^{-rT}\phi_T(v - (\alpha+1)i)}{\alpha^2 + \alpha - v^2 + i(2\alpha+1)v}$$

### 2. Discretization for FFT
Using Simpson's rule with an integration step size $\eta$ and a log-strike grid spacing of $\lambda$, we map the continuous inverse transform onto a discrete array structure that mirrors a Discrete Fourier Transform (DFT) summation layout:

$$C_T(k_u) \approx \frac{e^{-\alpha k_u}}{\pi}\sum_{j=1}^{N}e^{-i\frac{2\pi}{N}(j-1)(u-1)}e^{ibv_j}\psi_T(v_j)\frac{\eta}{3}\left[3+(-1)^j-\delta_{j-1}\right]$$

To achieve valid structural alignment for the fast algorithm, the grid parameters must satisfy the constraint:
$$\lambda\eta = \frac{2\pi}{N}$$

### 3. Recovering Put Prices
Once call prices $C_T(k)$ are available across the full strike grid from the transform above, put prices are recovered without any further Fourier work, via the standard no-arbitrage put-call parity identity:

$$P_T(K) = C_T(K) - S_0e^{-qT} + Ke^{-rT}$$

This identity is model-independent — it holds for VG, Black-Scholes, or any arbitrage-free pricing model, since it follows purely from the payoff structure rather than any distributional assumption.

---

## 📌 Tech Stack & Dependencies

* **Language:** Python 3.8+
* **Core Numerics:** NumPy, SciPy (`scipy.fft.fft`, `scipy.stats.norm`, `scipy.optimize.brentq`)
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
   cd Option-Pricing-FFT
   ```
2. Launch the local Streamlit server:
   ```bash
   streamlit run Pricing_Engine.py
   ```
3. Open your browser to the local URL (usually `http://localhost:8501`) to interact with the parameters.

## 📌 Sample Insights & Experimentation

By adjusting the **Kurtosis ($\nu$)** and **Skewness ($\theta$)** sliders in the sidebar, you can instantly see how the Variance Gamma model handles non-normal asset return profiles:
* Setting $\nu \to 0$ and $\theta \to 0$ aligns the pricing engine back toward log-normal, Black-Scholes-consistent behavior — a useful sanity check, and the one used to validate the bug fixes described above.
* Adjusting $\theta < 0$ tilts the pricing curve and the implied volatility smile, visually highlighting where Black-Scholes' flat-volatility assumption diverges from a skewed, fat-tailed return model.
* Toggling between **Call** and **Put** shows the same underlying VG return distribution priced through both payoff structures, related exactly by put-call parity.

---