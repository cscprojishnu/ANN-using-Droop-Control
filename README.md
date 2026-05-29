# ANN-using-Droop

# ⚡ Voltage Stability Enhancement in Microgrids: An ANN-Based Droop Control Approach

This repository contains the implementation and documentation of our research titled:

> **Voltage Stability Enhancement in Microgrids: An ANN-Based Droop Control Approach**  
> 🧠 Authors: [Jishnu Teja Dandamudi](mailto:djishnuteja2006@gmail.com), [Rupa Kandula](mailto:rupakandula21@gmail.com)  
> 🎓 Institution: Amrita School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore, India
> 🗓️ Year: 2025
> Published in: *CRC Press, Taylor & Francis Group, UK* (2026)  
> DOI: [10.1201/9781003740100-94](https://www.taylorfrancis.com/chapters/edit/10.1201/9781003740100-94) 

---

## 🔍 Abstract

This project presents an **Artificial Neural Network (ANN)**-based droop control strategy for voltage stability optimization in microgrids. Unlike conventional droop control, which uses static coefficients, this adaptive system leverages ANN to tune droop coefficients dynamically based on real-time power changes. The methodology combines:
- **State-Space Modeling**
- **PI Control & Virtual Inertia**
- **Adaptive Droop Coefficients**
- **ANN-based forecasting for voltage (V) and frequency (F)**

Implemented in **Python**, the model achieves superior voltage and frequency regulation under varying load conditions.

---

## 🧠 Key Features

- ✅ Adaptive droop control equations
- ✅ State-space modeling with matrix-based simulation
- ✅ Multi-layer ANN for voltage and frequency prediction
- ✅ Comparison with traditional and optimized droop models
- ✅ Visual analysis: Voltage/Current/Frequency state evolution

---

## 🛠 Methodology Overview

### 🔄 Droop Control Equations
```math
V = 230 - (mp + kp * (P - 1000)) * (P - 1000)
```
```math
F = 50  - (nq + kq * (Q - 500)) * (Q - 500)
```
