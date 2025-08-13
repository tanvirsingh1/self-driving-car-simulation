import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/driving_log.csv", header=None, nrows=2678)

# Convert column 3 to numeric, coercing errors to NaN
df[3] = pd.to_numeric(df[3], errors='coerce')

# Drop NaNs for clean plotting
df_steering = df[3].dropna()

bin_size = 0.1  # bin width
min_val = df_steering.min()
max_val = df_steering.max()

bins = np.arange(min_val, max_val + bin_size, bin_size)

plt.hist(df_steering, bins=bins, edgecolor='black')
plt.title('Steering Angle Distribution')
plt.xlabel('Steering Angle')
plt.ylabel('Frequency')
plt.show()
