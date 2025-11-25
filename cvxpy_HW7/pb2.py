import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Time settings
T = 10
N = T - 1   # 9 velocity steps

# Decision variables
x = cp.Variable(T)
v = cp.Variable(N)
u = cp.Variable(N)     # |v|
f = cp.Variable(N)     # fuel

constraints = []

# Initial position
constraints += [x[0] == 0]

# Dynamics
for t in range(N):
    constraints += [x[t+1] == x[t] + v[t]]

# Final position
constraints += [x[T-1] == 12]

# Absolute value constraints: u >= |v|
constraints += [u >= v, u >= -v]

# Fuel constraints: f >= max(u, 3u - 2)
constraints += [f >= u, f >= 3*u - 2]

# Objective
objective = cp.Minimize(cp.sum(f))

# Solve
problem = cp.Problem(objective, constraints)
problem.solve()

# Print results
# --- Position ---
plt.figure()
plt.plot(range(1, T+1), x.value, marker='o')
plt.title("Position vs Time")
plt.xlabel("Time")
plt.ylabel("Position")
plt.grid(True)

# --- Velocity ---
plt.figure()
plt.stem(range(1, N+1), v.value)
plt.title("Velocity vs Time")
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.grid(True)

# --- Fuel cost ---
plt.figure()
plt.stem(range(1, N+1), f.value)
plt.title("Fuel Cost per Time Step")
plt.xlabel("Time")
plt.ylabel("Fuel")
plt.grid(True)

plt.show()
