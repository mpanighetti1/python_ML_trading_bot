######################## Imports #############################

# Python Imports
import time
from tqdm import tqdm
import warnings

import pandas as pd
import numpy as np
import datetime
from collections import namedtuple

# Numba Imports
from numba import njit
from numba import prange

# Plotting Imports
import plotly.graph_objects as go
from plotly import offline

# Machine Learning Imports
import optuna as opt

# Trading Imports
import talib
import vectorbtpro as vbt

# Optuna Visualizations
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

##################### Data Gathering #########################

# Define column names
column_names = ['Gmt time', 'Open', 'High', 'Low', 'Close', 'Volume']

# Fetch Locally Saved Data (1 minute)
m1_data = pd.read_csv('data/Dukascopy/EURUSD_Candlestick_1_M_BID_07.01.2023-07.07.2023.csv', 
                      delimiter=',',  # Data is comma-separated
                      names=column_names,  # Add column names
                      skiprows=1)  # Skip the first row (column names)

# Convert 'Gmt time' to datetime and set it as index
m1_data['Gmt time'] = pd.to_datetime(m1_data['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')
m1_data.set_index('Gmt time', inplace=True)

# Ensure index is a DatetimeIndex
m1_data.index = pd.DatetimeIndex(m1_data.index.values)

# Rename index
m1_data.index.name = 'Open time'

# Set data for vectorbt
m1_data = vbt.Data.from_data({'EURUSD': m1_data})

## 1m data
m1_close_raw = m1_data.get('Close')

## 1m data
m1_open_raw  = m1_data.get('Open')
m1_high_raw  = m1_data.get('High')
m1_low_raw   = m1_data.get('Low')
m1_close_raw = m1_data.get('Close')

# Continuous indexing for 1m data
m1_open  = m1_open_raw.resample('1T').interpolate()
m1_high  = m1_high_raw.resample('1T').interpolate()
m1_low   = m1_low_raw.resample('1T').interpolate()
m1_close = m1_close_raw.resample('1T').interpolate()

# Continuous indexing for 1h data
h1_open = m1_open.resample('60T').first()
h1_high = m1_high.resample('60T').max()
h1_low = m1_low.resample('60T').min()
h1_close = m1_close.resample('60T').last()

##################### Ensure 2-Dim Arrays #######################

def ensure_2d(arr):
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr

##################### Indicator Functions #######################

####### SuperTrend Indicator Helper Functions ###################
def st_get_basic_bands(high, low, close, st_multiplier, st_period):
    st_medprice = vbt.talib('MEDPRICE').run(high, low, skipna=True).real.to_numpy().astype(np.float64)
    st_atr = vbt.talib('ATR').run(high, low, close, timeperiod=st_period, skipna=True).real.to_numpy().astype(np.float64)
    st_basic_upper, st_basic_lower = st_get_basic_bands_nb(st_atr, st_medprice, st_multiplier, st_period)
    return st_basic_upper, st_basic_lower

@njit(nogil=True)
def st_get_basic_bands_nb(st_atr, st_medprice, st_multiplier, st_period):
    st_matr = st_multiplier * st_atr
    st_basic_upper = st_medprice + st_matr
    st_basic_lower = st_medprice - st_matr
    return st_basic_upper, st_basic_lower

@njit(nogil=True)
def st_get_final_bands(close, st_basic_upper, st_basic_lower): 
    
    st_trend = np.full(close.shape, np.nan)  
    st_dir = np.full(close.shape, 1)
    st_long = np.full(close.shape, np.nan)
    st_short = np.full(close.shape, np.nan)
    
    for i in range(1, close.shape[0]):
        if close[i] > st_basic_upper[i - 1]:
            st_dir[i] = 1
        elif close[i] < st_basic_lower[i - 1]:
            st_dir[i] = -1
        else:
            st_dir[i] = st_dir[i - 1]
            if st_dir[i] > 0 and st_basic_lower[i] < st_basic_lower[i - 1]:
                st_basic_lower[i] = st_basic_lower[i - 1]
            if st_dir[i] < 0 and st_basic_upper[i] > st_basic_upper[i - 1]:
                st_basic_upper[i] = st_basic_upper[i - 1]
        if st_dir[i] > 0:
            st_trend[i] = st_long[i] = st_basic_lower[i]
        else:
            st_trend[i] = st_short[i] = st_basic_upper[i]    
    return st_trend, st_dir, st_long, st_short 

######################### SuperTrend Indicator ############################

expr = """
SuperTrend[st]:

# SuperTrend Indicator
basic_upper, basic_lower = st_get_basic_bands(@in_h1_high, @in_h1_low, @in_h1_close, @p_st_multiplier, @p_st_period)
trend, dir, long, short = st_get_final_bands(@in_h1_close, basic_upper, basic_lower)

# Returns
trend, dir, long, short

"""

SuperTrend = vbt.IF.from_expr(
    expr,
    takes_1d=True,

    # SuperTrend Indicator
    st_period=7,
    st_multiplier=5,
    st_get_basic_bands=st_get_basic_bands,
    st_get_final_bands=st_get_final_bands,
)

#################### Clean Signals ############################

@njit(nogil=True, parallel=True)
def clean_signals_nb(m1_close_np, st_dir_m1):  
    
    # Initialize signal arrays
    long_entry = np.full_like(m1_close_np, False, dtype=np.bool_)
    long_exit = np.full_like(m1_close_np, False, dtype=np.bool_)
    
    short_entry = np.full_like(m1_close_np, False, dtype=np.bool_)
    short_exit = np.full_like(m1_close_np, False, dtype=np.bool_)
    
    for col in prange(m1_close_np.shape[1]):

        for i in range(m1_close_np.shape[0]):

            ######################## LONGS ############################

            # Long Entries
            if (st_dir_m1[i, col] == 1.0 and st_dir_m1[i - 1, col] == -1.0):
                short_exit[i, col] = True
                long_entry[i, col] = True

            ######################## SHORTS ############################
            
            # Short Entries
            if (st_dir_m1[i, col] == -1.0 and st_dir_m1[i - 1, col] == 1.0):
                long_exit[i, col] = True
                short_entry[i, col] = True
           
            
    # Returns
    return long_entry, long_exit, short_entry, short_exit

######################### Objective Function ############################

def objective(trial, m1_close, h1_high,  h1_low,  h1_close):
    
    ######################### Indicator Calls ############################
    ######################### Paramaterization ########################### 
        
    st_multiplier = trial.suggest_float("st_multiplier", 1.0, 20.0, step=0.1) 
    st_period = trial.suggest_int("st_period", 2, 1000)

    st = SuperTrend.run(h1_high, h1_low, h1_close, # calls h1_high, h1_low, h1_close
                        st_multiplier=st_multiplier,
                        st_period=st_period,                        
                        execute_kwargs=dict(                                        
                                        engine='dask',
                                        chunk_len='auto',
                                        show_progress=True
                                        ))

    ######################## Resample Everything To 1 Min ##########################

    # Use some placeholder value
    placeholder = -99999

    #################### Nan to Placeholder ########################

    st_dir = st.dir.fillna(placeholder)
    st_long = st.long.fillna(placeholder)
    st_short = st.short.fillna(placeholder)

    #################### Reindex To 1 Min ##########################

    st_dir_m1 = st_dir.reindex(m1_close.index).ffill()
    st_dir_m1 = st_dir_m1.bfill()
    st_long_m1 = st_long.reindex(m1_close.index).ffill()
    st_long_m1 = st_long_m1.bfill()
    st_short_m1 = st_short.reindex(m1_close.index).ffill()
    st_short_m1 = st_short_m1.bfill()

    #################### Placeholder To NaN ########################

    st_dir_m1 = st_dir_m1.replace(placeholder, np.nan)
    st_short_m1 = st_short_m1.replace(placeholder, np.nan)
    st_long_m1 = st_long_m1.replace(placeholder, np.nan)    

    #################### Convert to Numpy ##########################

    st_dir_m1 = st_dir_m1.to_numpy()

    #################### 1 Minute OHLC Data ########################

    m1_close_np = m1_close.to_numpy()

    ################### Ensure 2-Dim Array #########################

    st_dir_m1 = ensure_2d(st_dir_m1)
    m1_close_np = ensure_2d(m1_close_np)
    
    #################### Clean Signals Call ############################

    long_entry, \
    long_exit, \
    short_entry, \
    short_exit = clean_signals_nb(m1_close_np, st_dir_m1)
    
    ######################### Portfolio Wrapper #########################

    pf = vbt.Portfolio.from_signals(
        close=m1_close,
        entries=long_entry,
        exits=long_exit,
        short_entries=short_entry,
        short_exits=short_exit,
        fees=0.00005,
        slippage=0.0000269,
        init_cash=15000,
        freq='1m',
    )

    return pf.total_return # Return as Objective


######################### Outer Function Closure ###########################

def outer_objective(m1_close, h1_high,  h1_low,  h1_close):
    
    def call_objective(trial):
        return objective(trial, m1_close, h1_high, h1_low, h1_close)
        
    return call_objective

###########################################################################
############################### Main Program ##############################
###########################################################################

if __name__ == '__main__':       
    
    #start_time = time.perf_counter() # Start Timer
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    ######################### Optuna Optimization #########################
    
    # Store the stduy in a database
    # Local: 
    # storage = "sqlite:///supertrend_param"
    storage = "sqlite:///super_trend_op.db"
    study_name = "super_trend_op"
    
    random_sampler = opt.samplers.RandomSampler()
    initial_sampler = opt.samplers.QMCSampler(warn_independent_sampling=False)
    optmize_sampler = opt.samplers.TPESampler(n_startup_trials=1000)

    # Maximize the objective value 
    study = opt.create_study(study_name=study_name,
                            storage=storage,
                            direction="maximize",
                            sampler=optmize_sampler,
                            load_if_exists=True)
    
    study.optimize(outer_objective(m1_close, h1_high,  h1_low,  h1_close), n_trials=int(1e6))

    print("Best trial: ")
    best = study.best_params

    print(" Value: ", study.best_value)

    print(" Params: ")
    for key, value in best.items():
        print(" {}: {}".format(key, value))

    plot_optimization_history(study)
        
    ######################### Save Portfolio ############################

    # Get the current date and time
    #now = datetime.datetime.now()

    # Convert the datetime object to a string in the format YYYY-MM-DD
    #date_string = now.strftime("%Y-%m-%d")

    # Use this string to name your file
    #pf.save(f'portfolios/pf_{date_string}')

    ######################### Performance Time ##########################
    '''
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print()
    print(f"The program took {elapsed_time} seconds to complete.")
    ''' 
    pass