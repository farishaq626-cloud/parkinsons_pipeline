import numpy as np
from statsmodels.stats.power import TTestIndPower

# Methodology constants agreed upon for your paper
effect_size = 0.5
alpha = 0.05
power = 0.8

# Calculate sample size
analysis = TTestIndPower()
n_required = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1.0)

print(f"--- Statistical Roadmap ---")
print(f"Target sample size: {int(np.ceil(n_required))} patients total (approx. {int(np.ceil(n_required)/2)} per group).")