import numpy as np
import pandas as pd

# Load dataset
file_path = r"C:\Users\cscpr\Desktop\SEM2\PROJECTS\EEE\final\SGSMA_Competiton 2024_PMU_DATA\PMU_Data_with_Anomalies and Events\Bus2_Competition_Data.csv" # Update this if needed
df = pd.read_csv(file_path)

# Extract relevant data
df_filtered = df[['TIMESTAMP', 'BUS2_VA_MAG', 'BUS2_VB_MAG', 'BUS2_VC_MAG', 'BUS2_Freq', 'Event']]

# Compute average voltage magnitude
df_filtered['Voltage_Avg'] = df_filtered[['BUS2_VA_MAG', 'BUS2_VB_MAG', 'BUS2_VC_MAG']].mean(axis=1)

# Define nominal values
V_nom = 230  # Nominal voltage per phase (V)
P_ref = 1000  # Reference active power (W)
Q_ref = 500   # Reference reactive power (Var)

# Compute deviations
df_filtered['Voltage_Dev'] = df_filtered['Voltage_Avg'] - V_nom
df_filtered['Freq_Dev'] = df_filtered['BUS2_Freq'] - 60  # Assuming 60Hz system

# Compute average droop slopes
m_p_avg = np.mean(df_filtered['Freq_Dev'] / P_ref)  # P-f droop coefficient
n_q_avg = np.mean(df_filtered['Voltage_Dev'] / Q_ref)  # Q-V droop coefficient

# Save learned parameters
output_file = "learned_droop_params.npy"
np.save(output_file, {'m_p': m_p_avg, 'n_q': n_q_avg})

print(f"Learned parameters saved to {output_file}")
