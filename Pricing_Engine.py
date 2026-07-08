import streamlit as st
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.stats import norm

# =====================================================================
# Core Variance Gamma Pricing Engine
# =====================================================================
# Computes the analytical frequency-domain characteristic function for the Variance Gamma asset price model.
def vg_characteristic_function(u, S0, r, q, T, sigma, nu, theta):
    omega = (1.0 / nu) * np.log(1.0 - theta * nu - 0.5 * (sigma**2) * nu)
    drift = np.log(S0) + (r - q + omega) * T
    phi = np.exp(1j * u * drift) * (1.0 - 1j * theta * nu * u + 0.5 * (sigma**2) * nu * (u**2))**(-T / nu)
    return phi

# Generates the reciprocal log-strike and frequency vectors required for the discrete FFT algorithm.
def setup_numerical_grids(S0, N, eta):
    lambda_grid = (2 * np.pi) / (N * eta)
    j = np.arange(N)
    v_j = j * eta
    b = np.log(S0) - (N * lambda_grid) / 2
    k_m = b + j * lambda_grid
    strikes = np.exp(k_m)
    return strikes, v_j, b

# Constructs the damped Carr-Madan modified characteristic function integrand to ensure square integrability.
def build_fourier_integrand(v_j, S0, r, q, T, sigma, nu, theta, alpha):
    u = v_j - (alpha + 1) * 1j
    phi = vg_characteristic_function(u, S0, r, q, T, sigma, nu, theta)
    denominator = (alpha**2 + alpha - v_j**2) + 1j * (2 * alpha + 1) * v_j
    psi = (np.exp(-r * T) * phi) / denominator
    return psi

# Applies Simpson's rule weights and executes the Fast Fourier Transform to compute raw frequency-domain option values.
def execute_fft_transformation(psi, v_j, b, eta, N):
    weights = np.ones(N)
    weights[0] = 1/3
    weights[1::2] = 4/3
    weights[2::2] = 2/3
    weights[-1] = 1/3
    fft_input = np.exp(-1j * b * v_j) * psi * eta * weights
    fft_output = np.real(fft(fft_input))
    return fft_output

# Removes the analytical damping factor and reconstructs accurate in-the-money call prices using put-call parity.
def finalize_option_prices(fft_output, strikes, S0, r, q, T, alpha):
    N = len(fft_output)
    call_prices = np.zeros(N)
    k_m = np.log(strikes)
    raw_prices = (np.exp(-alpha * k_m) / np.pi) * fft_output
    
    for i in range(N):
        K = strikes[i]
        if K >= S0:
            call_prices[i] = raw_prices[i]
        else:
            otm_put_price = raw_prices[i]
            call_prices[i] = otm_put_price + S0 * np.exp(-q * T) - K * np.exp(-r * T)
            call_prices[i] = max(call_prices[i], S0 * np.exp(-q * T) - K * np.exp(-r * T))
            
    return call_prices

# Orchestrates the complete five-step pipeline from raw market inputs to final arrays of strikes and option prices.
def complete_fft_pricing_engine(S0, r, q, T, sigma, nu, theta, alpha=1.5, N=4096, eta=0.25):
    strikes, v_j, b = setup_numerical_grids(S0, N, eta)
    psi = build_fourier_integrand(v_j, S0, r, q, T, sigma, nu, theta, alpha)
    fft_output = execute_fft_transformation(psi, v_j, b, eta, N)
    call_prices = finalize_option_prices(fft_output, strikes, S0, r, q, T, alpha)
    return strikes, call_prices


# Calculates the benchmark European call option price using the classical Black-Scholes-Merton analytical formula.
def black_scholes_call(S0, K, r, q, T, sigma_bs):
        # Handle edge case for zero time or zero volatility to avoid division by zero
    if T <= 0 or sigma_bs <= 0:
        return np.maximum(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma_bs**2) * T) / (sigma_bs * np.sqrt(T))
    d2 = d1 - sigma_bs * np.sqrt(T)
    
    call_price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# =====================================================================
# Streamlit UI Configuration
# =====================================================================

st.set_page_config(layout="wide", page_title="FFT vs Black-Scholes Experiment")

st.title("Comparison Variance Gamma FFT vs Black-Scholes")
st.write("Comparing a fat-tailed, skewed asset pricing engine against the log-normal benchmark.")

# Two-column layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Market Inputs")
    
    st.subheader("Market Environment")
    S0 = st.number_input("Stock Spot Price ($S_0$)", min_value=1.0, value=100.0, step=1.0)
    T = st.slider("Time to Maturity ($T$ in Years)", min_value=0.05, max_value=3.0, value=0.5, step=0.05)
    r = st.slider("Risk-Free Rate ($r$)", min_value=0.0, max_value=0.15, value=0.05, step=0.01, format="%.2f")
    q = st.slider("Dividend Yield ($q$)", min_value=0.0, max_value=0.10, value=0.00, step=0.01, format="%.2f")
    
    st.subheader("Variance Gamma (VG) Parameters")
    sigma = st.slider("Base Volatility ($\sigma$)", min_value=0.05, max_value=0.60, value=0.20, step=0.01)
    nu = st.slider("Kurtosis / Fat Tails ($\\nu$)", min_value=0.01, max_value=0.80, value=0.25, step=0.01)
    theta = st.slider("Skewness / Asymmetry ($\\theta$)", min_value=-0.60, max_value=0.00, value=-0.15, step=0.01)

    st.subheader("Grid Settings")
    N = st.selectbox("FFT Grid Nodes ($N$)", options=[2048, 4096, 8192], index=1)
    eta = st.slider("Frequency Grid Step ($\eta$)", min_value=0.05, max_value=0.50, value=0.25, step=0.05)

# ---------------------------------------------------------------------
# EXECUTION PIPELINE
# ---------------------------------------------------------------------

# 1. Running step-by-step VG FFT Pricing Engine
grid_strikes, vg_prices = complete_fft_pricing_engine(S0, r, q, T, sigma, nu, theta, alpha=1.5, N=N, eta=eta)

# 2. Computing the mathematically matched Black-Scholes Volatility
bs_vol = np.sqrt(sigma**2 + (theta**2) * nu)

# 3. Calculating the Black-Scholes prices across the exact same strike grid
bs_prices = np.array([black_scholes_call(S0, K, r, q, T, bs_vol) for K in grid_strikes])

# 4. Consolidating results into a unified Data Frame for Streamlit plotting
df_experiment = pd.DataFrame({
    "Strike Price (K)": grid_strikes,
    "Variance Gamma (FFT)": vg_prices,
    "Black-Scholes-Merton": bs_prices,
    "Absolute Discrepancy ($)": np.abs(vg_prices - bs_prices)
})

# Filtering out extreme wings to keep chart clean 
lower_bound = S0 * 0.7
upper_bound = S0 * 1.3
df_filtered = df_experiment[(df_experiment["Strike Price (K)"] >= lower_bound) & (df_experiment["Strike Price (K)"] <= upper_bound)]

# ---------------------------------------------------------------------
# UI RENDERING SIDE
# ---------------------------------------------------------------------
with col2:
    st.header("Results")
    
    m1, m2 = st.columns(2)
    m1.metric(label="Calculated Equivalent BS Volatility", value=f"{bs_vol*100:.2f}%")
    m2.metric(label="Max Pricing Divergence in Window", value=f"${df_filtered['Absolute Discrepancy ($)'].max():.2f}")
    
    st.write("---")
    
    # Plotting both curves together on the same graph
    st.subheader("Visual Curve Overlay: VG Engine vs. Black-Scholes")
    # Streamlit handles multiple line series automatically if they are in columns next to each other
    st.line_chart(
        data=df_filtered, 
        x="Strike Price (K)", 
        y=["Variance Gamma (FFT)", "Black-Scholes-Merton"]
    )
    
    # Displaying the underlying raw comparison data grid
    st.subheader("Data Comparison Array")
    st.dataframe(
        df_filtered.style.format({
            "Strike Price (K)": "{:.2f}", 
            "Variance Gamma (FFT)": "${:.2f}", 
            "Black-Scholes-Merton": "${:.2f}",
            "Absolute Discrepancy ($)": "${:.2f}"
        }),
        use_container_width=True,
        height=250
    )