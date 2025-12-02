# Intermittent Wells Optimisation  
**Algorithm for optimisation of intermittent wells (ПКВ-mode) through analytical modelling and OLGA-verified simulation**

This repository contains a Python implementation of the algorithm developed as part of my final qualification thesis:  
**“Optimization of the "reservoir-well" system using modeling methods in software.”**

The project provides a reproducible engineering workflow for selecting optimal accumulation and pumping cycle durations for wells equipped with ESPs (УЭЦН), operating in intermittent mode (ПКВ).

---

## Overview

Intermittent operation of ESP wells is widely used on mature fields with decreasing reservoir energy and rising watercut. However, selecting cycle durations empirically (the usual field practice) often leads to:

- unstable ESP operation;  
- excessive number of restarts - reduced MTBF;  
- non-optimal liquid production;  
- high energy consumption (low URE/УРЭ).

This repository presents a **unified optimisation approach**, combining:

1. **Unsteady well modelling in OLGA**  
   – adapted to telemetry and PVT data  
   – includes ESP, inflow, and downhole gas separator models  

2. **Analytical algorithm**  
   – derives dynamic behaviour during accumulation & pumping  
   – solves ODEs for bottom-hole pressure & inflow  
   – accounts for ESP ramp-up/ramp-down  
   – evaluates production per cycle and energy efficiency  

3. **Final optimisation and verification**  
   – optimal cycles are validated through the OLGA simulation  
   – production uplift and ESP load reduction are quantified  

---

## Features

- Fully reproducible analytical workflow for intermittent cycle optimisation  
- Physically justified mathematical model (linear inflow + dynamic BHP)  
- Support for user-defined:  
  - productivity index (PI)  
  - reservoir pressure  
  - well geometry  
  - ESP nominal capacity  
  - ramp-up time  
  - boundaries of cycle durations  
- Automatic search for optimal cycle durations  
- Visualisation of:  
  - accumulated production  
  - ESP duty ratio  
  - 4-cycle dynamic behaviour (pressure, inflow, rates)  
- Implemented in pure Python 3.12 (NumPy, Matplotlib, Tkinter)  

