import streamlit as st
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.stats import norm

# Function to calculate the value of the specific equation
def specific_equation_value(rp, alpha, epsilon, N, qc):
    denominator = (1/np.sqrt(N) * np.sqrt((1 + epsilon * rp) * (1 - (1 + epsilon * rp) * qc) / alpha + (1 - qc) / (1 - alpha)))
    if denominator == 0:
        # Handle the divide by zero scenario
        return np.inf  # or any other appropriate handling
    else:
        value = epsilon * rp / denominator
    return value

# Function to find combinations of alpha and rp
def find_specific_fraction(epsilon, rp, alpha_range, N, qc, confidence):
    target_value = norm.ppf(confidence)
    combinations = []
    for alpha in alpha_range:
        if np.isclose(specific_equation_value(rp, alpha, epsilon, N, qc), target_value, atol=0.01):  # Allowing a small tolerance
            combinations.append((rp, alpha, confidence))
    return combinations


# Function to find combinations of alpha and rp
def find_specific_combinations(epsilon, rp_range, alpha_range, N, qc, confidence):
    target_value = norm.ppf(confidence)
    combinations = []
    for rp in rp_range:
        for alpha in alpha_range:
            if np.isclose(specific_equation_value(rp, alpha, epsilon, N, qc), target_value, atol=0.01):  # Allowing a small tolerance
                combinations.append((rp, alpha, confidence))
    return combinations

# confidence values 
confidence_options = [0.75, 0.8, 0.9, 0.95]
selected_confidences = st.multiselect('Select confidence intervals:', confidence_options, default=[0.95])

# Use Streamlit's functions to create inputs
N = st.number_input('Monthly traffic:', min_value=1, value=10000)
qc = st.number_input('Baseline conversion (%):', min_value=0.001, max_value=100.0, value=1.0) / 100
epsilon = st.number_input('Elasticity (% change in conversion given 1 % change in price):', min_value=0.0, value=1.0)
days = st.number_input('Number of days:', min_value=0, value=15)
rp = st.number_input('Price difference:', min_value=0.0, value=0.1)

# Example ranges for alpha and rp (adjust as needed)
rp_range = np.linspace(0, 0.3, 200)  # Range of r_p values
alpha_range = np.linspace(0.001, 0.5, 200)  # Range of alpha values

combinations_specific = []

# Find combinations
for confidence in selected_confidences:
    tmp = find_specific_combinations(epsilon, rp_range, alpha_range, int(N * days / 30), qc, confidence)
    combinations_specific.extend(tmp)

df = pd.DataFrame(combinations_specific, columns=['price_difference', 'fraction', 'target_value'])


fig, ax = plt.subplots(1, 1, figsize=(20, 10))

for target_value, tmp in df.groupby('target_value'):
    tmp.plot(x='price_difference', y='fraction', ax=ax, label=round(target_value, 2))

ax.set_xlabel('price difference')
ax.set_ylabel('treatment fraction')
plt.title(f'Price difference vs Fraction Treatment given {days} days, baseline conversion = {qc}, epsilon = {epsilon}')

# Display the calculated value
specific_value = find_specific_fraction(epsilon, rp, np.linspace(0.001, 0.5, 1000) , int(N * (days / 30)), qc, selected_confidences[0])

# Calculate the value
value_to_display = round(specific_value[0][1] if len(specific_value) > 0 else 0, 2)

# Format the value as a percentage
percentage_value = f"{value_to_display * 100}%"  # Multiplying by 100 to convert to percentage

# Display using st.metric
st.metric(label=f"Fraction of traffic required for price difference of {int(rp * 100)}% over a period of {days} days", value=percentage_value)

# Use Streamlit to display outputs
st.pyplot(fig)
