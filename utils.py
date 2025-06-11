
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys


#####################    DATA FUNCTIONS      ###############################
from scipy.signal import resample
from scipy.signal import butter, filtfilt
def resample_signal(signal,time, target_fs):

    step_size = np.diff(time)
    dt = np.mean(step_size)

    original_fs = 1/dt #Sampling Frequency
    print("Approx sampling frequency", original_fs)

    # Compute the resampling ratio
    resampling_ratio = target_fs / original_fs

    # Determine the new length of the resampled signal
    new_length = int(len(signal) * resampling_ratio)

    # Resample the signal
    resampled_signal = resample(signal, new_length)

    return resampled_signal;

def resample_dataframe(df, target_fs):

    step_size = np.diff(df['time'])
    dt = np.mean(step_size)

    original_fs = 1/dt #Sampling Frequency

    resampled_df = pd.DataFrame()
    for column in df.columns:
        if column != 'time':
            resampled_df[column] = resample_signal(df[column],df['time'] , target_fs)
            
    
    length = len(resampled_df)
    start_time = df['time'].iloc[0]
    end_time = df['time'].iloc[-1]
    resampled_df['time'] = np.linspace(start_time, end_time, length)       

    return resampled_df

#Filtering the signal
def signal_filtering(cut_off,order,signal,time):
    step_size = np.diff(time)
    dt = np.mean(step_size)

    fs = 1/dt #Sampling Frequency
    print("Approx sampling frequency",fs)

    cutoff_freq = cut_off  # Cutoff frequency of the low-pass filter in Hz
    nyquist_freq = 0.5 * fs  # Nyquist frequency (half of the sampling rate)
    cutoff_norm = cutoff_freq / nyquist_freq  # Normalize the cutoff frequency

    # Design the low-pass filter using a Butterworth filter
    b, a = butter(order, cutoff_norm, btype='low')

    # Apply the filter to the signal using filtfilt (zero-phase filtering)
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal


def rms(vec):
    sq = np.square(vec)
    sq = np.square(vec)
    rms = np.sqrt(np.mean(sq))

    return rms



############ OTHER UTILS    #################
def liter_per_minute_to_kg_per_s(lpm):
    """Convert flow rate from liters per minute to kg per second."""
    return (lpm) / 60  # assuming water density ~1000 kg/m^3

def air_prandtl_number(T_celsius):
    """
    Calculate the air Prandtl number at a given temperature in Celsius.
    
    Parameters:
    T_celsius (float): Temperature in Celsius (°C)
    
    Returns:
    float: Prandtl number
    """
    # Convert Celsius to Kelvin
    T = T_celsius + 273.15
    
    # Dynamic viscosity of air (μ) in kg/(m·s)
    # Approximation using Sutherland's formula
    C1 = 1.458e-6
    S = 110.4
    mu = C1 * (T ** 1.5) / (T + S)
    
    # Specific heat at constant pressure (Cp) in J/(kg·K)
    # Approximate polynomial fit for air
    Cp = 1005 + (T - 273.15) * 0.1  # Rough linear approximation
    
    # Thermal conductivity of air (k) in W/(m·K)
    # Approximate polynomial fit for air
    k = 0.024 + (T - 273.15) * 0.000075  # Rough linear approximation
    
    # Prandtl number (Pr)
    Pr = mu * Cp / k
    
    return Pr
        
def prandtl_number(mu, Cp, k):
    """
    Calculate the Prandtl number for a fluid.
    
    Parameters:
    mu (float): Dynamic viscosity (kg/(m·s))
    Cp (float): Specific heat at constant pressure (J/(kg·K))
    k (float): Thermal conductivity (W/(m·K))
    
    Returns:
    float: Prandtl number
    """
    Pr = mu * Cp / k
    return Pr

def nussetl_number(Reynolds,Prandtl,Hydraulic_Diam,pipe_lenght):
    Re = Reynolds
    Pr = Prandtl
    D = Hydraulic_Diam
    l = pipe_lenght

    Nu = 0.023*(Re**0.8)*(Pr**0.8)*(1+D/l)**0.07
    return Nu

def nusselt_number_dittus_boelter(Re, Pr, heating=True):
    n = 0.4 if heating else 0.3
    return 0.023 * (Re ** 0.8) * (Pr ** n)


def reynolds_number(V, D_h, nu):
    """    
    Parameters:
    V (float): Fluid velocity (m/s)
    D_h (float): Hydraulic diameter (m)
    nu (float): Kinematic viscosity of the fluid (m^2/s)
    
    Returns:
    float: Reynolds number
    """
    Re = (V * D_h) / nu
    return Re
    