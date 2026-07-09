import streamlit as st
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.stats import norm
from scipy.optimize import brentq

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
    fft_input = np.exp(-1j * b * v_j) * psi * eta * weights
    fft_output = np.real(fft(fft_input))
    return fft_output

# Removes the analytical damping factor to recover call prices directly from the damped Fourier transform.
def finalize_option_prices(fft_output, strikes, S0, r, q, T, alpha):
    k_m = np.log(strikes)
    # With alpha > 0 the Carr-Madan damped transform already yields the call price
    # itself (not an OTM put) at every strike, ITM and OTM alike, so no put-call
    # parity conversion is needed here.
    call_prices = (np.exp(-alpha * k_m) / np.pi) * fft_output
    call_prices = np.maximum(call_prices, S0 * np.exp(-q * T) - strikes * np.exp(-r * T))
    call_prices = np.maximum(call_prices, 0.0)
    return call_prices

# Derives put prices from call prices via put-call parity: Put = Call - S0*e^(-qT) + K*e^(-rT).
# This is exact and model-agnostic (holds for VG, Black-Scholes, or any European pricing model),
# so once the call price is right, no separate Fourier inversion is needed for puts.
def put_from_call(call_prices, strikes, S0, r, q, T):
    put_prices = call_prices - S0 * np.exp(-q * T) + strikes * np.exp(-r * T)
    put_intrinsic = np.maximum(strikes * np.exp(-r * T) - S0 * np.exp(-q * T), 0.0)
    put_prices = np.maximum(put_prices, put_intrinsic)
    put_prices = np.maximum(put_prices, 0.0)
    return put_prices

# Orchestrates the complete pipeline from raw market inputs to final arrays of strikes, call prices, and put prices.
@st.cache_data
def complete_fft_pricing_engine(S0, r, q, T, sigma, nu, theta, alpha=1.5, N=4096, eta=0.25):
    strikes, v_j, b = setup_numerical_grids(S0, N, eta)
    psi = build_fourier_integrand(v_j, S0, r, q, T, sigma, nu, theta, alpha)
    fft_output = execute_fft_transformation(psi, v_j, b, eta, N)
    call_prices = finalize_option_prices(fft_output, strikes, S0, r, q, T, alpha)
    put_prices = put_from_call(call_prices, strikes, S0, r, q, T)
    return strikes, call_prices, put_prices


# Calculates the benchmark European call option price using the classical Black-Scholes-Merton analytical formula.
def black_scholes_call(S0, K, r, q, T, sigma_bs):
        # Handle edge case for zero time or zero volatility to avoid division by zero
    if T <= 0 or sigma_bs <= 0:
        return np.maximum(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma_bs**2) * T) / (sigma_bs * np.sqrt(T))
    d2 = d1 - sigma_bs * np.sqrt(T)
    
    call_price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Calculates the benchmark European put option price via put-call parity on the Black-Scholes call formula.
def black_scholes_put(S0, K, r, q, T, sigma_bs):
    call_price = black_scholes_call(S0, K, r, q, T, sigma_bs)
    put_price = call_price - S0 * np.exp(-q * T) + K * np.exp(-r * T)
    return max(put_price, 0.0)

# Inverts the Black-Scholes formula via root-finding to back out the implied volatility a given price implies.
def implied_volatility(price, S0, K, r, q, T, option_type="call"):
    if option_type == "call":
        pricer = black_scholes_call
        intrinsic = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    else:
        pricer = black_scholes_put
        intrinsic = max(K * np.exp(-r * T) - S0 * np.exp(-q * T), 0.0)
    if price <= intrinsic + 1e-8:
        return np.nan
    try:
        return brentq(lambda s: pricer(S0, K, r, q, T, s) - price, 1e-6, 5.0)
    except ValueError:
        return np.nan

# Computes the annualized skewness and excess kurtosis of VG log-returns over horizon T from the process cumulants.
def vg_moments(sigma, nu, theta, T):
    c2 = (sigma**2 + theta**2 * nu) * T
    c3 = (2 * theta**3 * nu**2 + 3 * sigma**2 * theta * nu) * T
    c4 = 3 * (sigma**4 * nu + 2 * theta**4 * nu**3 + 4 * sigma**2 * theta**2 * nu**2) * T
    skewness = c3 / c2**1.5
    excess_kurtosis = c4 / c2**2
    return skewness, excess_kurtosis

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

    with st.expander("How this works", expanded=False):
        st.markdown(
            "Standard Black-Scholes assumes log-normal, symmetric returns. The "
            "**Variance Gamma (VG)** model replaces that with a Brownian motion "
            "time-changed by a Gamma process, which lets it produce skewed, "
            "fat-tailed return distributions closer to what's observed in real "
            "markets. Its prices are computed via the **Carr-Madan FFT method**: "
            "a damping factor makes the transform square-integrable, Simpson's "
            "rule discretizes the pricing integral, and an FFT evaluates prices "
            "across the whole strike grid at once in O(N log N) instead of "
            "pricing each strike one at a time in O(N²)."
        )

    st.subheader("Market Environment")
    option_type = st.radio("Option Type", options=["Call", "Put"], horizontal=True)
    S0 = st.number_input("Stock Spot Price ($S_0$)", min_value=1.0, value=100.0, step=1.0)
    T = st.slider("Time to Maturity ($T$ in Years)", min_value=0.05, max_value=3.0, value=0.5, step=0.05)
    r = st.slider("Risk-Free Rate ($r$)", min_value=0.0, max_value=0.15, value=0.05, step=0.01, format="%.2f")
    q = st.slider("Dividend Yield ($q$)", min_value=0.0, max_value=0.10, value=0.00, step=0.01, format="%.2f")
    
    st.subheader("Variance Gamma (VG) Parameters")
    sigma = st.slider("Base Volatility ($\\sigma$)", min_value=0.05, max_value=0.60, value=0.20, step=0.01)
    nu = st.slider("Kurtosis / Fat Tails ($\\nu$)", min_value=0.01, max_value=0.80, value=0.25, step=0.01)
    theta = st.slider("Skewness / Asymmetry ($\\theta$)", min_value=-0.60, max_value=0.00, value=-0.15, step=0.01)

    st.subheader("Grid Settings")
    N = st.selectbox("FFT Grid Nodes ($N$)", options=[2048, 4096, 8192], index=1)
    eta = st.slider("Frequency Grid Step ($\\eta$)", min_value=0.05, max_value=0.50, value=0.25, step=0.05)

    st.subheader("Display Range")
    strike_range_pct = st.slider(
        "Strike Window (% of $S_0$)", min_value=0.05, max_value=0.9, value=0.3, step=0.05,
        help="Controls how wide a band around the spot price is shown in the charts/table below."
    )

# ---------------------------------------------------------------------
# EXECUTION PIPELINE
# ---------------------------------------------------------------------

# 1. Running step-by-step VG FFT Pricing Engine
grid_strikes, vg_call_prices, vg_put_prices = complete_fft_pricing_engine(S0, r, q, T, sigma, nu, theta, alpha=1.5, N=N, eta=eta)
vg_prices = vg_call_prices if option_type == "Call" else vg_put_prices

# 2. Computing the mathematically matched Black-Scholes Volatility
bs_vol = np.sqrt(sigma**2 + (theta**2) * nu)

# 3. Calculating the Black-Scholes prices across the exact same strike grid
bs_pricer = black_scholes_call if option_type == "Call" else black_scholes_put
bs_prices = np.array([bs_pricer(S0, K, r, q, T, bs_vol) for K in grid_strikes])

# 4. Consolidating results into a unified Data Frame for Streamlit plotting
df_experiment = pd.DataFrame({
    "Strike Price (K)": grid_strikes,
    f"Variance Gamma (FFT) {option_type}": vg_prices,
    f"Black-Scholes-Merton {option_type}": bs_prices,
    "Absolute Discrepancy ($)": np.abs(vg_prices - bs_prices)
})

# Filtering out extreme wings to keep chart clean 
lower_bound = S0 * (1 - strike_range_pct)
upper_bound = S0 * (1 + strike_range_pct)
df_filtered = df_experiment[(df_experiment["Strike Price (K)"] >= lower_bound) & (df_experiment["Strike Price (K)"] <= upper_bound)]

# 5. Backing out the implied volatility smile the VG prices imply, versus the flat BS input vol
df_filtered = df_filtered.copy()
df_filtered["VG Implied Vol (%)"] = [
    implied_volatility(p, S0, K, r, q, T, option_type.lower()) * 100
    for p, K in zip(df_filtered[f"Variance Gamma (FFT) {option_type}"], df_filtered["Strike Price (K)"])
]
df_filtered["BS Input Vol (%)"] = bs_vol * 100

# 6. Computing the annualized skewness / excess kurtosis the chosen VG parameters imply
vg_skew, vg_kurt = vg_moments(sigma, nu, theta, T)

# ---------------------------------------------------------------------
# UI RENDERING SIDE
# ---------------------------------------------------------------------
with col2:
    st.header(f"Results — {option_type} Option")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label="Equivalent BS Volatility", value=f"{bs_vol*100:.2f}%")
    m2.metric(label="Max Divergence in Window", value=f"${df_filtered['Absolute Discrepancy ($)'].max():.2f}")
    m3.metric(label="VG Return Skewness", value=f"{vg_skew:.3f}", help="0 = symmetric. Negative means a fatter left (crash) tail, matching equity index skew.")
    m4.metric(label="VG Excess Kurtosis", value=f"{vg_kurt:.3f}", help="0 = normal tails. Positive means fatter tails than log-normal Black-Scholes assumes.")
    
    st.write("---")
    
    # Plotting both curves together on the same graph
    st.subheader(f"Visual Curve Overlay: VG Engine vs. Black-Scholes ({option_type})")
    # Streamlit handles multiple line series automatically if they are in columns next to each other
    st.line_chart(
        data=df_filtered, 
        x="Strike Price (K)", 
        y=[f"Variance Gamma (FFT) {option_type}", f"Black-Scholes-Merton {option_type}"]
    )
    
    # Plotting the implied volatility smile the VG prices produce vs. the flat BS input vol
    st.subheader("Implied Volatility Smile: VG vs. Flat Black-Scholes")
    st.line_chart(
        data=df_filtered,
        x="Strike Price (K)",
        y=["VG Implied Vol (%)", "BS Input Vol (%)"]
    )

    st.write("---")

    # Displaying the underlying raw comparison data grid
    st.subheader("Data Comparison Array")
    st.dataframe(
        df_filtered.style.format({
            "Strike Price (K)": "{:.2f}", 
            f"Variance Gamma (FFT) {option_type}": "${:.2f}", 
            f"Black-Scholes-Merton {option_type}": "${:.2f}",
            "Absolute Discrepancy ($)": "${:.2f}",
            "VG Implied Vol (%)": "{:.2f}%",
            "BS Input Vol (%)": "{:.2f}%"
        }),
        use_container_width=True,
        height=250
    )

    st.download_button(
        label="Download data as CSV",
        data=df_filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"vg_vs_bs_{option_type.lower()}_pricing_comparison.csv",
        mime="text/csv"
    )