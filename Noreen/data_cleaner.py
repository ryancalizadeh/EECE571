import numpy as np
import pandas as pd

def data_cleaner(data):
    """
    Filters the input data based on time segments to retain multiples of 0.02s.
    Preserves the input format (Pandas DataFrame or NumPy Array) and all columns.
    
    Parameters:
    data: Input data (DataFrame or 2D Array). 
          The first column (index 0) MUST be the time column.
    
    Returns:
    Filtered data in the same format as the input.
    """
    # 1. Extract the time column for logic, handling both DataFrame and Array
    if isinstance(data, pd.DataFrame):
        t = data.iloc[:, 0].values # Convert col 0 to numpy array for speed
    elif hasattr(data, 'ndim') and data.ndim > 1:
        t = data[:, 0]             # Use first column of numpy array
    else:
        t = data                   # Fallback for 1D input
        
    # 2. Define masks for each segment
    # (Handling triplets at boundaries by assigning them to the 'Keep 3rd' groups)
    
    # 0s - 5s (Keep 3rd): Includes 5.00 triplet
    mask_0_5   = (t <= 5.01)
    
    # 5s - 6s (Keep All): Strictly between 5.00 and 6.00 triplets
    mask_5_6   = (t > 5.01) & (t < 5.99)
    
    # 6s - 11s (Keep 3rd): Includes 6.00 and 11.00 triplets
    mask_6_11  = (t >= 5.99) & (t < 11.01)
    
    # 11s - 12s (Keep All): Strictly between 11.00 and 12.00 triplets
    mask_11_12 = (t > 11.01) & (t < 11.99)
    
    # 12s - 17s (Keep 3rd): Includes 12.00 and 17.00 triplets
    mask_12_17 = (t >= 11.99) & (t < 17.01)
    
    # 17s - 20s (Keep All): After 17.00 triplet
    mask_17_20 = (t > 17.01)

    # 3. Get indices for each segment
    idx_0_5   = np.where(mask_0_5)[0]
    idx_5_6   = np.where(mask_5_6)[0]
    idx_6_11  = np.where(mask_6_11)[0]
    idx_11_12 = np.where(mask_11_12)[0]
    idx_12_17 = np.where(mask_12_17)[0]
    idx_17_20 = np.where(mask_17_20)[0]

    # 4. Apply filters (Keep every 3rd vs Keep all)
    sel_0_5   = idx_0_5[::3]
    sel_5_6   = idx_5_6
    sel_6_11  = idx_6_11[::3]
    sel_11_12 = idx_11_12
    sel_12_17 = idx_12_17[::3]
    sel_17_20 = idx_17_20

    # 5. Combine indices
    all_indices = np.concatenate([
        sel_0_5, sel_5_6, sel_6_11, sel_11_12, sel_12_17, sel_17_20
    ])
    all_indices.sort()

    # 6. Return the result in the ORIGINAL format
    if isinstance(data, pd.DataFrame):
        return data.iloc[all_indices]  # Return DataFrame with all columns
    elif data.ndim == 2:
        return data[all_indices, :]    # Return 2D Array with all columns
    else:
        return data[all_indices]       # Return 1D Array


def input_signal(P0, A, t, t0, tf):
    # # 1. Extract time column
    # if isinstance(data, pd.DataFrame):
    #     t = data.iloc[:, 0].values
    # elif hasattr(data, 'ndim') and data.ndim > 1:
    #     t = data[:, 0]
    # else:
    #     t = np.array(data)

    values = np.zeros_like(t)

    # 2. Define Masks
    mask_before = t < t0
    mask_after = t > tf
    mask_between = (t >= t0) & (t <= tf)

    # 3. Apply "Before" Logic
    values[mask_before] = P0

    # 4. Apply "After" Logic
    # Holds the value of sin(tf)
    values[mask_after] = A + A*np.sin(2*np.pi  * 0.5 * (tf-t0)**2 / 5)

    # 5. Apply "Between" Logic (Step Function)
    if np.any(mask_between):
        t_subset = t[mask_between]
        step = 0.02
        tolerance = 1e-5
        
        # Find nearest multiple of 0.02
        k = np.round(t_subset / step)
        t_nearest = k * step
        
        # Determine effective time (hold previous if not close enough to next)
        effective_t = np.where(t_subset >= t_nearest - tolerance, t_nearest, t_nearest - step)
        
        values[mask_between] = A + A*np.sin(2*np.pi  * 0.5 * (effective_t-t0)**2 / 5)

    return values