import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import scipy.io

# Load the data
mat_data = scipy.io.loadmat(r"C:\Users\sidat\Downloads\Global_temp_data.mat")
data = mat_data['Temperature_file']

# Extract years and temperature changes
years = data[:, 0]
temp_change = data[:, 1]

print(f"Data span: {years[0]:.0f} - {years[-1]:.0f}")
print(f"Number of data points: {len(years)}")

# Create the design matrix A
n = len(years)
A = np.column_stack([np.ones(n), years, years**2])

print("\nDesign matrix A shape:", A.shape)
print("First few rows of A:")
print(A[:3])

x = cp.Variable(3)
b = temp_change

# Objective: minimize the sum of squared residuals
objective = cp.Minimize(cp.sum_squares(A @ x - b))

# Solve the problem
problem = cp.Problem(objective)
problem.solve()

# Extract the optimal coefficients
x1_opt = x.value[0]
x2_opt = x.value[1]
x3_opt = x.value[2]

print("\n" + "="*60)
print("OPTIMAL COEFFICIENTS:")
print("="*60)
print(f"x1 (intercept):     {x1_opt:.6f}")
print(f"x2 (linear term):   {x2_opt:.6f}")
print(f"x3 (quadratic term): {x3_opt:.9f}")
print("="*60)

# Model equation
print(f"\nQuadratic Model:")
print(f"Temp_change = {x1_opt:.6f} + {x2_opt:.6f}*Year + {x3_opt:.9f}*Year²")

# Calculate residuals and R-squared
y_pred = A @ x.value
residuals = temp_change - y_pred
ss_res = np.sum(residuals**2)
ss_tot = np.sum((temp_change - np.mean(temp_change))**2)
r_squared = 1 - (ss_res / ss_tot)

print(f"\nModel Statistics:")
print(f"R-squared: {r_squared:.6f}")
print(f"RMSE: {np.sqrt(np.mean(residuals**2)):.6f} °C")

# Predict temperature change for year 2030
year_2030 = 2030
temp_2030 = x1_opt + x2_opt * year_2030 + x3_opt * (year_2030**2)

print("\n" + "="*60)
print(f"PREDICTION FOR YEAR 2030:")
print("="*60)
print(f"Predicted temperature change: {temp_2030:.4f} °C")
print("="*60)

# Create the plot
plt.figure(figsize=(12, 8))

# Plot original data
plt.scatter(years, temp_change, alpha=0.6, s=30, label='Observed Data', color='blue')

# Plot the fitted model for years 1880-2040
years_extended = np.linspace(1880, 2040, 500)
temp_fitted = x1_opt + x2_opt * years_extended + x3_opt * (years_extended**2)

plt.plot(years_extended, temp_fitted, 'r-', linewidth=2, label='Quadratic Model Fit')

# Highlight the prediction for 2030
plt.scatter([year_2030], [temp_2030], color='green', s=200, marker='*', 
            edgecolors='black', linewidth=1.5, label=f'2030 Prediction: {temp_2030:.4f}°C', zorder=5)

# Add vertical line at 2030
plt.axvline(x=year_2030, color='green', linestyle='--', alpha=0.5, linewidth=1)

plt.xlabel('Years', fontsize=12, fontweight='bold')
plt.ylabel('Change in temperature (°C)', fontsize=12, fontweight='bold')
plt.title('Global Change in Temperature\nQuadratic Regression Model', fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(1880, 2040)

# Add text box with model equation
textstr = f'Model: T = {x1_opt:.2f} + {x2_opt:.4f}Y + {x3_opt:.6f}Y²\n$R^2$ = {r_squared:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('temperature_quadratic_model.png', dpi=300, bbox_inches='tight')
print("\n✓ Plot saved to outputs folder")

plt.close()

