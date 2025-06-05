
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from scipy.signal import butter, filtfilt, resample
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

#Constant Cell values
cp_cell = 855                   #J/Kg*K
m_cell = 0.070                  #Kg
CellAH = 4.5                    #AH

cp_water = 4185     # J/kg⋅K (specific heat capacity)
cp_air = 1005  # J/kg⋅K (specific heat capacity)
cp_pk = 2850                    #J*KgK

rho_PK = 0.2        #W/m*K
rho_paste = 1;      #W/m*K
rho_water = 1000    # kg/m³ (density)

#Corrective coefficient Cell internal resistance for heat
R_corr = 1.7



#####################    DATA FUNCTIONS      ###############################
def read_csv_to_dataframe(folder_name, file_name, what):
    # Construct the full file path

    if what == "cooling":
        file_path = os.path.join('VadenaLog', '2024_04_08', folder_name, 'parsed', 'primary', file_name)
        df = pd.read_csv(file_path)
        #Time conditioning
        df['time'] = (df['_timestamp'] - df.at[0,'_timestamp'])/1e6


        
    elif what == "temperature":
        file_path = os.path.join('VadenaLog', '2024_04_08', folder_name, 'parsed', 'secondary', file_name)
        df = pd.read_csv(file_path)
        #Temperatures Offset
        T_offset = 2

        #Time conditioning
        df['time'] = (df['_timestamp'] - df.at[0,'_timestamp'])/1e6
        df['T_air'] = df['top_left']
        df['T_rad_in'] = df['top_right']-T_offset
        df['T_rad_out'] = df['bottom_right']
    
    elif what == "speed":
        file_path = os.path.join('VadenaLog', '2024_04_08', folder_name, 'parsed', 'secondary', 'angular_velocity.csv')
        df = pd.read_csv(file_path)
        #Time conditioning
        df['time'] = (df['_timestamp'] - df.at[0,'_timestamp'])/1e6
        df['mean_speed'] = ((df['fl']+df['fr'])/2)*(0.4064/2)*3.6 #to rad/s->km/h

    elif what == "current":
        file_path = os.path.join('VadenaLog', '2024_04_08', folder_name, 'parsed', 'primary', 'hv_current.csv')
        df = pd.read_csv(file_path)
        df['time'] = (df['_timestamp'] - df.at[0,'_timestamp'])/1e6

    elif what == "current_endurance":
        file_path = os.path.join('2023_08_20_Endurances','2023_08_20_Endurances', folder_name, 'parsed', 'primary', 'hv_current.csv')
        df = pd.read_csv(file_path)
        df['time'] = (df['_timestamp'] - df.at[0,'_timestamp'])/1e6
        
        
    else:
        print("what field do not match")
        return 0
    
    return df;

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

#Build internal Resistance Map
def CellInternalResistance():
    CellData = pd.read_excel('MolicellP45B_Parameter.xlsx')
    soc = CellData['SOC2'][CellData['SOC2']!=0]
    R0_T1 = CellData['R0_T1'][0:len(soc)]
    R0_T2 = CellData['R0_T2'][0:len(soc)]


    df = pd.DataFrame()
    df['SOC'] = soc
    df['R0_20'] = R0_T1
    df['R0_45'] = R0_T2

    temp = [df['R0_20'],df['R0_45']]
    R_map = interp2d(soc,[20,45],temp)

    temp_range = np.linspace(15,70,11)

    return R_map


def rms(vec):
    sq = np.square(vec)
    sq = np.square(vec)
    rms = np.sqrt(np.mean(sq))

    return rms




############ OTHER UTILS    #################

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

def NTC_RtoC(R):
    beta = 3354
    R0 = 10000
    T0 = 25+273.15

    
    one_o_T = (1/T0) + (1/beta)*np.log(R/R0)

    T = 1/(one_o_T) - 273.15

    return T









######################  COLING FUNCTIONS  #########################

def radiator_steady_state(air_massflow,T_air_in,Q,q_dot_water):
    #q_dot_water in l/min
    #air_massflow in kg/s


    # Thermodynamic properties
    epsilon = 0.43      #Radiator Effectivness 

    Cmin = cp_air*air_massflow
    T_water_in = T_air_in + Q/(epsilon*Cmin)

    m_dot_water = q_dot_water/60
    T_water_out = T_water_in - Q/(m_dot_water*cp_water)

    return T_water_out, T_water_in


def Water_Cell_Delta_T(Q_cell,T_water):
    #Cooling Snake parameter
    
    htc_water = 500     #W/m^2K
    s_snake = 1*1e-3    #m
    s_paste = 0.2*1e-3  #m
    A_cell_Snake = (3370)*1e-6

    Rth_snake = s_snake/(A_cell_Snake*rho_PK)
    #print('Snake Thermal Resistance:',Rth_snake,"K/W")

    Rth_paste = s_paste/(A_cell_Snake*rho_paste)
    #print('PAste Thermal Resistance:',Rth_paste,"K/W")

    Rth_water = 1/(A_cell_Snake*htc_water)
    #print("Convective Thermal Resistance:",Rth_water,"K/W")

    R_th = Rth_paste+Rth_water+Rth_snake
    print("Total thermal Resistance between Cell and Water:",R_th,"K/W")

    Cell_Water_DeltaT = R_th*Q_cell
    T_cell = T_water+Cell_Water_DeltaT

    return T_cell


def cell_snake_temperatures_v2(Tin_water_up,Tin_water_down,Tcell_prev,Cell_Q,dt):
    #Qcs = (Tcell_prev-T_snake_prev)/Rth_paste   #Cell-Snake heat

    #Volume Flow
    q_dot_pack = 12                 #l/min
    q_dot_segment = q_dot_pack/6    #l/min
    m_dot_pack = q_dot_pack/60      #kg/s
    m_dot_segment = q_dot_segment/60#kg/s
    m_dot_snake = m_dot_segment/5   #kg/s

    #Thermal conductivity
    htc_water = 500                 #W/m^2K

    #Geometry
    s_snake = 1*1e-3                    #m
    s_paste = 0.2*1e-3                  #m
    A_cell_Snake_half = (1685)*1e-6     #m^2

    #Cell-Water Thermal resistence
    Rth_snake = s_snake/(A_cell_Snake_half*rho_PK)
    Rth_paste = s_paste/(A_cell_Snake_half*rho_paste)
    Rth_water = 1/(A_cell_Snake_half*htc_water)
    R_th = Rth_paste+Rth_water+Rth_snake


    #Cell-Water heat transfer, in the 2 half of cell
    Qcw_up = (Tcell_prev-Tin_water_up)/R_th 
    Qcw_down = (Tcell_prev-Tin_water_down)/R_th

    #Water temperature after heat transfer
    Tout_water_up = Tin_water_up + Qcw_up/(m_dot_snake*cp_water)
    Tout_water_down = Tin_water_down + Qcw_down/(m_dot_snake*cp_water)

    #Cell temperature
    Tcell_after = Tcell_prev + ((Cell_Q-Qcw_up-Qcw_down) / (m_cell*cp_cell))*dt

    return Tout_water_up,Tout_water_down,Tcell_after


def single_cell_thermal_simulation(time,Cell_Current,m_dot_air,q_dot_pack, sel_R,rad_model,T0,Tair):
    Ttot = time[len(time)-1]
    print("Total time= ", Ttot)
    n = len(time)
    print("N simulation step", n)

    #Geometry
    A_rad_water = 0.2               #m^2
    A_rad_air = 1.2                 #m^2

    #Heat Capacities
    cp_radiator = 902               #J/Kg*K

    #Mass
    m_can = 3.5                     #kg
    m_snake = 0.172                 #Kg
    m_radiator = 1.5                #kg

    #Volume Flow
    q_dot_segment = q_dot_pack/6    #l/min
    m_dot_pack = q_dot_pack/60      #kg/s
    m_dot_segment = q_dot_segment/60#kg/s
    m_dot_snake = m_dot_segment/5   #kg/s

    #Thermal conductivity
    htc_water = 500                 #W/m^2K
    epsilon_radiator = 0.42         #[-]
    htc_water_rad = 5570            #W/m^2K
    htc_air_rad = 131               #W/m^2K

    # Electrical Proprieties
    R0 = 0.016                      #Ohm

    R_map = CellInternalResistance()

    #Initial Conditions
    T0_water = T0
    T0_cell = T0
    T0_air = Tair
    T0_snake = T0
    T0_can = T0
    T0_rad = T0
    SOC0 = 1

    #Change some gemoetrical chatacteristics w.r.t before, now considere 1 single cell at the time
    #Geometry
    s_snake = 1*1e-3                    #m
    s_paste = 0.2*1e-3                  #m
    A_cell_Snake_half = (1685)*1e-6     #m^2

    #Cell-Water Thermal resistence
    Rth_snake = s_snake/(A_cell_Snake_half*rho_PK)
    Rth_paste = s_paste/(A_cell_Snake_half*rho_paste)
    Rth_water = 1/(A_cell_Snake_half*htc_water)
    R_th = Rth_paste+Rth_water+Rth_snake

    #Water Air thermal Resistence
    Rth_water_rad = 1/(A_rad_water*htc_water_rad)
    Rth_air_rad = 1/(A_rad_air*htc_air_rad)
    Rth_rad = Rth_water_rad + Rth_air_rad

    #Initialization
    #Cell_Current = np.ones(n)*I_cell

    Twater = np.ones((31,n))*T0_water
    T_water_in_segment = np.ones(n)*T0_water
    T_water_in_radiator = np.ones(n)*T0_water
    Tcell = np.ones((15,n))*T0_cell
    T_air = np.ones(n)*T0_air
    T_water_in_can = np.ones(n)*T0_can
    T_can = np.ones(n)*T0_can
    T_rad = np.ones(n)*T0_rad

    Q_radiator = np.zeros(n)
    Cell_Q = np.zeros(n)
    Qwr = np.zeros(n)
    Qra = np.zeros(n)
    Q_in_rad = np.zeros(n)
    #
    SOC = np.ones(n)*SOC0
    R_cell = np.ones(n)*R0

    for i in range(len(time)-1):
        Cmin = m_dot_air*cp_air
        dt = max([time[i+1]-time[i],1e-3])
        #print("dt = ",dt)
        #SoC integration
        SOC[i+1] = SOC[i] - Cell_Current[i]*dt/(CellAH*3600)

        final_time = 0
        if SOC[i+1]<0:
                final_time = time[i]
                final_idx = i
                print("Final time:",time[i])
                break

        #print("SOC = ",SOC)
        
        

        #Cell Resistance
        if sel_R == 1:
            R_cell[i] = R_map(SOC[i+1],Tcell[14,i]) * R_corr
            
        elif sel_R == 0:
            R_cell[i] = R0
        else:
            print('Invalid Resistance selection')
            break


        Cell_Q[i]= ((Cell_Current[i]**2)*R_cell[i])
        #print("Cell_Q=", Cell_Q[i])

        Twater[0,i+1] =  T_water_in_segment[i]

        for j in range(0,15):
            if j!=14:
                Twater[j+1,i+1], Twater[30-j,i+1], Tcell[j,i+1] = cell_snake_temperatures_v2(Twater[j,i],Twater[30-j-1,i],Tcell[j,i],Cell_Q[i],dt)
            elif(j==14):
                Qcw_up = (Tcell[14,i]-Twater[14,i])/R_th 
                Qcw_down = (Tcell[14,i]-Twater[15,i])/R_th

                #Water temperature after heat transfer
                Twater[15,i+1] = Twater[14,i] + Qcw_up/(m_dot_snake*cp_water)
                Twater[16,i+1] = Twater[15,i+1] + Qcw_down/(m_dot_snake*cp_water)
                
                #Cell temperature
                Tcell[14,i+1] = Tcell[14,i] + ((Cell_Q[i]-Qcw_up-Qcw_down) / (m_cell*cp_cell))*dt
            else:
                print('Invalid Radiator selection')

        #Out of last cell is the input temperature of the radiator
        T_water_in_radiator[i+1] = Twater[30,i+1]
        
        #RADIATOR
        if rad_model == 0:
        #First Radiator Version
            Q_radiator[i+1] = Cmin*(T_water_in_radiator[i+1]-T_air[i])*epsilon_radiator
            T_water_in_can[i+1] = T_water_in_radiator[i+1] - Q_radiator[i+1]/(m_dot_pack*cp_water)
        elif rad_model ==1:
        #Second version radiator
            Qwr[i+1] = (T_water_in_radiator[i+1]-T_rad[i])/Rth_water_rad
            Qra[i+1] = (T_rad[i] - T_air[i]) / Rth_air_rad
            Q_in_rad[i+1]  = Qwr[i+1] - Qra[i+1]
            T_rad[i+1] = T_rad[i] + (Q_in_rad[i+1]/(m_radiator*cp_radiator))*dt
            T_water_in_can[i+1] = T_water_in_radiator[i+1] - Qwr[i+1]/(m_dot_pack*cp_water)
        else:
            print()



        #Catch Can
        #1/dt
        T_can[i+1] = (m_dot_pack*T_water_in_can[i+1] + m_can*T_can[i]/dt) * (dt/(m_can + m_dot_pack*dt))
        T_water_in_segment[i+1] = T_can[i+1]


    return Tcell,T_water_in_radiator,T_water_in_segment,Cell_Q,SOC


def only_cell_inertia(time,Cell_Current, sel_R,T0):
    Ttot = time[len(time)-1]
    print("Total time= ", Ttot)
    n = len(time)
    print("N simulation step", n)

    # Electrical Proprieties
    SOC0 = 1
    R0 = 0.016                      #Ohm
    R_map = CellInternalResistance()

    Cell_Q = np.zeros(n)
    SOC = np.ones(n)*SOC0
    R_cell = np.ones(n)*R0
    Tcell = np.ones(n)*T0


    for i in range(len(time)-1):
        dt = time[i+1]-time[i]
        #print("dt = ",dt)
        #SoC integration
        SOC[i+1] = SOC[i] - Cell_Current[i]*dt/(CellAH*3600)

        final_time = 0
        if SOC[i+1]<0:
                final_time = time[i]
                final_idx = i
                print("Final time:",time[i])
                break
        
        

        #Cell Resistance

        if sel_R == 1:
            R_cell[i] = R_map(SOC[i+1],Tcell[i]) * R_corr

            #print("SOC = ",SOC)
        elif sel_R == 0:
            R_cell[i] = R0
        else:
            print('Invalid Resistance selection')
            break


        Cell_Q[i]= ((Cell_Current[i]**2)*R_cell[i])
        Tcell[i+1] = Tcell[i] + (Cell_Q[i]/(m_cell*cp_cell))*dt

    return Tcell, Cell_Q


def free_air_single_cell_temp(time,Cell_Current, sel_R,T0_cell,htc_air,Tamb,SOC_0):
    Ttot = time[len(time)-1]
    print("Total time= ", Ttot)
    n = len(time)
    print("N simulation step", n)

    # Electrical Proprieties
    SOC0 = SOC_0
    R0 = 0.016                      #Ohm
    R_map = CellInternalResistance()

    Cell_A = 0.021*np.pi*0.070
    Rth_air = 1/(Cell_A*htc_air)

    Cell_Q = np.zeros(n)
    Q_tot = np.zeros(n)
    Air_Q = np.zeros(n)
    SOC = np.ones(n)*SOC0
    R_cell = np.ones(n)*R0
    Tcell = np.ones(n)*T0_cell
    T_amb = np.ones(n)*Tamb
  

    for i in range(len(time)-1):
        dt = time[i+1]-time[i]
        #print("dt = ",dt)
        #SoC integration
        SOC[i+1] = SOC[i] - Cell_Current[i]*dt/(CellAH*3600)

        final_time = 0
        if SOC[i+1]<0:
                final_time = time[i]
                final_idx = i
                print("Final time:",time[i])
                break
        
        

        #Cell Resistance

        if sel_R == 1:
            R_cell[i] = R_map(SOC[i+1],Tcell[i]) * R_corr

            #print("SOC = ",SOC)
        elif sel_R == 0:
            R_cell[i] = R0
        else:
            print('Invalid Resistance selection')
            break


        Q_tot[i]= ((Cell_Current[i]**2)*(R_cell[i]))
        Air_Q[i] = (Tcell[i]-T_amb[i])/Rth_air
        Cell_Q[i] = Q_tot[i]-Air_Q[i]

        Tcell[i+1] = Tcell[i] + (Cell_Q[i]/(m_cell*cp_cell))*dt

    return Tcell, Cell_Q, Air_Q, SOC, R_cell


def aircooling__thermal_simulation(time,Cell_Current,air_m_dot,htc_air, sel_R,T0_cell,T0_air):
    Ttot = time[len(time)-1]
    print("Total time= ", Ttot)
    n = len(time)
    print("N simulation step", n)

    # Electrical Proprieties
    R0 = 0.016                      #Ohm

    R_map = CellInternalResistance()

    #Initial Conditions
    SOC0 = 1

    #Change some gemoetrical chatacteristics w.r.t before, now considere 1 single cell at the time
    #Geometry

    #Cell-Water Thermal resistence
    h_Cell_Air = 54.410*1e-3
    A_cell_Air = ((np.pi*(21*1e-3)))*h_Cell_Air
    
    Rth_air = 1/(A_cell_Air*htc_air)


    Tcell = np.ones((6,n))*T0_cell
    Tair_pipe = np.ones((7,n))*T0_air

    Cell_Q = np.zeros(n)
    #
    SOC = np.ones(n)*SOC0
    R_cell = np.ones(n)*R0

    for i in range(len(time)-1):
        dt = max([time[i+1]-time[i],1e-3])
        #print("dt = ",dt)
        #SoC integration
        SOC[i+1] = SOC[i] - Cell_Current[i]*dt/(CellAH*3600)

        final_time = 0
        if SOC[i+1]<0:
                final_time = time[i]
                final_idx = i
                print("Final time:",time[i])
                break


        

        #Cell Resistance
        if sel_R == 1:
            R_cell[i] = R_map(SOC[i+1],Tcell[5,i])* R_corr
            #print("SOC = ",SOC)
        elif sel_R == 0:
            R_cell[i] = R0
        else:
            print('Invalid Resistance selection')
            break


        Cell_Q[i]= ((Cell_Current[i]**2)*R_cell[i])
        #print("Cell_Q=", Cell_Q[i])

        Tair_pipe[0,i+1] =  T0_air

        for j in range(0,6):
                #Cell to air heat
                Q_ca = (Tcell[j,i]-Tair_pipe[j,i])/Rth_air
                #In cell heat
                Q_c = Cell_Q[i]-Q_ca

                Tair_pipe[j+1,i+1] = Tair_pipe[j,i] + Q_ca/(air_m_dot*cp_air)
                
                #Cell temperature
                Tcell[j,i+1] = Tcell[j,i] + ((Q_c) / (m_cell*cp_cell))*dt

    return Tcell,Tair_pipe,Cell_Q,SOC











############ PUMP DATA ANALISYS ##################
#Read from CSV the logged data
def read_pump(path):
    df = pd.read_csv(path)
    dt = df['ms'][2]-df['ms'][1]
    df['time'] = np.linspace(0,dt*len(df['ms']),len(df['ms']))/1000
    df['P_in'] = df['CH1']
    df['P_out'] = df['CH2']

    return df

def pump_map_data(plot,data1,data2,data3,data4):
    if plot==1:
        plt.figure()
        plt.plot(data1['time'],data1['P_out']-data1['P_in'])
        plt.plot(data2['time'],data2['P_out']-data2['P_in'])
        plt.plot(data3['time'],data3['P_out']-data3['P_in'])
        plt.plot(data4['time'],data4['P_out']-data4['P_in'])
        plt.grid()
        plt.xlabel("time [s]")
        plt.ylabel("dP [mbar]")
        plt.legend(["Data1","Data2","Data3","Data4"])

    #Build data frame
    df = pd.DataFrame({'Pin':[np.mean(data1['P_in']),np.mean(data2['P_in']),np.mean(data3['P_in']),np.mean(data4['P_in'])],
                       'Pout':[np.mean(data1['P_out']),np.mean(data2['P_out']),np.mean(data3['P_out']),np.mean(data4['P_out'])],
                       'std':[np.std(data1['P_out']-data1['P_in']), np.std(data2['P_out']-data2['P_in']), np.std(data3['P_out']-data3['P_in']), np.std(data4['P_out']-data4['P_in'])]         
                       })
    df['dP'] = df['Pout']-df['Pin']

    return df