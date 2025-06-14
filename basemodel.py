import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Define system parameters
R = 0.1  # Further increased Line resistance (Ohm)
L = 0.3   # Further increased Line inductance (H)
C = 0.03  # Further increased Filter capacitance (F)
J = 0.05  # Virtual inertia constant
D_f = 0.02  # Frequency damping factor
V_nom = 230  # Nominal voltage (V)
P_ref = 1000  # Reference active power (W)
Q_ref = 500   # Reference reactive power (Var)

# Advanced Droop control coefficients (now dynamic)
def dynamic_droop(P, Q):
    m_p = 0.03 + 0.00001 * (P - 1000)  # Adaptive P-f droop
    n_q = 0.15 + 0.00002 * (Q - 500)  # Adaptive Q-V droop
    return m_p, n_q

# Expanded State-space model matrices with Q-V droop, virtual inertia, and PI control
A = np.array([[0, 1, 0, 0, 0],
              [-1/(L*C), -R/L, 1/L, 0, 0],
              [0, -1, -D_f/J, 0.05, 1/J],
              [0.02, 0, -0.05, -0.02, 0],
              [0, 0, 1, 0, -0.1]])  # Added PI control state
B = np.array([[0], [1/L], [0.5], [0.1], [0.05]])
C = np.array([[1, 0, 0, 0, 0]])
D = np.array([[0]])

# Create enhanced state-space system
sys = signal.StateSpace(A, B, C, D)

# Time vector
t = np.linspace(8, 10, 3000)  # Start plot from t=8s

# Input signal (multi-step change in load demand)
P_test = np.array([950, 1000, 1050, 1100])
Q_test = np.array([450, 500, 550, 600])
U = np.piecewise(t, [t < 2, (t >= 2) & (t < 6), (t >= 6) & (t < 8), t >= 8], 
                 [P_ref, P_ref + 200, P_ref - 150, P_ref + 300])  # Multi-step power variations

# Simulate system response
t_out, y, x = signal.lsim(sys, U, t)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(t_out, y, label='Voltage Response', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Highly Complex Microgrid Voltage Response with Advanced Droop Control (t>=8s)')
plt.legend()
plt.grid()
plt.show()

# Additional analysis: Plotting system states
plt.figure(figsize=(12, 6))
plt.plot(t_out, x[:, 0], label='Voltage State', linestyle='--')
plt.plot(t_out, x[:, 1], label='Current State', linestyle='-.')
plt.plot(t_out, x[:, 2], label='Frequency Deviation', linestyle=':')
plt.plot(t_out, x[:, 3], label='Control Feedback', linestyle='-')
plt.plot(t_out, x[:, 4], label='PI Controller Output', linestyle='-.')
plt.xlabel('Time (s)')
plt.ylabel('State Variables')
plt.title('System States Evolution in Highly Complex Microgrid (t>=8s)')
plt.legend()
plt.grid()
plt.show()