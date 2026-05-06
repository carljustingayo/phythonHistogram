import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual theme
sns.set_theme(style="whitegrid")

# 1. LOAD REAL KAGGLE DATA
try:
    df = pd.read_csv("StudentsPerformance.csv")
except FileNotFoundError:
    print("Error: 'StudentsPerformance.csv' not found. Download it via Kaggle CLI first.")
    exit()

# 2. PREPARE THE 10 SPECIFIC DISTRIBUTIONS
np.random.seed(42)
n_samples = 1000

data = {
    "1. Normal (Math Scores)": df["math score"],
    "2. Binomial (Gender)": df["gender"].map({'female': 1, 'male': 0}),
    "3. Exponential": np.random.exponential(scale=1.0, size=n_samples),
    "4. Poisson": np.random.poisson(lam=3, size=n_samples),
    "5. Triangular": np.random.triangular(left=0, mode=5, right=10, size=n_samples),
    "6. Lognormal": np.random.lognormal(mean=0, sigma=0.5, size=n_samples),
    "7. Gamma": np.random.gamma(shape=2, scale=1, size=n_samples),
    "8. Beta": np.random.beta(a=0.5, b=0.5, size=n_samples),
    "9. Weibull": np.random.weibull(a=1.5, size=n_samples),
    "10. Uniform": np.random.uniform(low=0, high=1, size=n_samples)
}

# 3. PLOT (5 rows x 2 columns for a clean 10-plot grid)
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(14, 20))
axes = axes.flatten()

# Colors for a nice aesthetic
colors = sns.color_palette("husl", 10)

for i, (name, values) in enumerate(data.items()):
    ax = axes[i]
    # Use discrete=True for Binomial and Poisson to keep bars centered on integers
    is_discrete = any(word in name.lower() for word in ["binomial", "poisson"])
    
    sns.histplot(values, kde=not is_discrete, ax=ax, color=colors[i], bins=30)
    ax.set_title(name, fontsize=14, fontweight='bold')
    ax.set_ylabel('')
    ax.set_xlabel('')

plt.tight_layout()
plt.show()