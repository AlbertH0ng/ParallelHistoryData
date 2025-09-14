import os
import pickle
import numpy as np
import pandas as pd
from arch import arch_model
import warnings
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')

# ===========================
# CONFIGURATION SETTINGS
# ===========================

# Paths configuration
raw_data_path = "/Users/antinghong/quantclass-data-folder/coin-binance-spot-swap-preprocess-pkl-1h" # 预处理数据路径
processed_data_path = "/Users/antinghong/Documents/LocalCode/ParallelHistoryData/Output" # 输出路径

# Custom generation configuration
GENERATION_MODES = ['GBM_Gravity', 'GARCH', 'Anomaly_Injection'] # 生成模式 # 'GBM_Gravity', 'GARCH',
GENERATION_COUNT = 1  # 每个模式生成世界数量
RANDOM_SEED = None  # 随机种子，设置为None则每次运行生成不同的世界

CUSTOM_PARAMETERS = {
    'GBM_Gravity': {
        'sigma_scale': 1.0, # 波动率系数
        'G': 0.6, # 引力系数
        'drift_scale': 1.0 # 漂移率缩放系数
    },
    'GARCH': {
        'sigma_scale': 1.0, # 波动率系数
        'G': 0.0, # 引力系数: 1.0=纯原始价格修改, 0.0=纯生成新价格, 0.5=混合
        'drift_scale': 1.0 # 漂移率缩放系数
    },
    'Anomaly_Injection': {
        'anomaly_prob': 0.02, # 异常概率
        'recovery_hours': 36 # 恢复小时数
    }
}

MAX_WORKERS = max(cpu_count()- 1, 1)  # Parallel processing workers

# ===========================
# DATA PROCESSING FUNCTIONS
# ===========================

# Load dict files
def load_data():
    with open(os.path.join(raw_data_path, 'spot_dict.pkl'), 'rb') as f:
        spot_dict = pickle.load(f)
    with open(os.path.join(raw_data_path, 'swap_dict.pkl'), 'rb') as f:
        swap_dict = pickle.load(f)
    return spot_dict, swap_dict

# Calculate VaR for all symbols
def calculate_var_dict(data_dict, confidence_level=0.95):
    """Calculate VaR for all symbols"""
    var_dict = {}
    for symbol, df in data_dict.items():
        valid_data = df.dropna(subset=['close'])
        if len(valid_data) > 50:
            returns = valid_data['close'].pct_change().dropna()
            if len(returns) > 0:
                var_value = np.percentile(np.abs(returns), confidence_level * 100)
                var_dict[symbol] = max(var_value, 0.005)  # Minimum 0.5% VaR
    return var_dict

# Function to adjust linked fields based on new OHLC
def adjust_linked_fields(df):
    # Update avg_price fields based on new OHLC (these might be used for vwap1m)
    if 'avg_price_1m' in df.columns:
        df['avg_price_1m'] = df['open']*0.8 + df['high']*0.05 + df['low']*0.05 + df['close']*0.1
    if 'avg_price_5m' in df.columns:
        df['avg_price_5m'] = df['open']*0.7 + df['high']*0.05 + df['low']*0.05 + df['close']*0.2
    
    # Adjust volume-related fields based on price changes
    if 'volume' in df.columns and 'quote_volume' in df.columns:
        # Calculate current average price for volume adjustment
        avg_price = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        # Keep the ratio but add some noise
        original_ratio = df['quote_volume'] / (df['volume'] * avg_price + 1e-8)
        df['quote_volume'] = df['volume'] * avg_price * original_ratio * (1 + np.random.normal(0, 0.01, len(df)))
    
    # Advanced adjustment for trading activity fields based on original ratios
    # This preserves market microstructure relationships while adapting to new prices
    
    # 1. Adjust trade_num based on original quote_volume/trade_num ratio
    if 'trade_num' in df.columns and 'quote_volume' in df.columns:
        # Calculate original average trade size (quote volume per trade)
        original_avg_trade_size = df['quote_volume'] / (df['trade_num'] + 1e-8)
        # Generate new trade_num based on new quote_volume and original ratio
        # Add small random variation (±5%) to simulate natural fluctuation
        trade_size_variation = 1 + np.random.normal(0, 0.05, len(df))
        df['trade_num'] = df['quote_volume'] / (original_avg_trade_size * trade_size_variation)
        df['trade_num'] = np.maximum(df['trade_num'], 1)  # Ensure at least 1 trade
    
    # 2. Adjust taker_buy volumes based on original market pressure ratios
    if 'taker_buy_quote_asset_volume' in df.columns and 'quote_volume' in df.columns:
        # Calculate original taker buy ratio (buy pressure indicator)
        original_buy_ratio = df['taker_buy_quote_asset_volume'] / (df['quote_volume'] + 1e-8)
        # Apply ratio to new quote_volume with small random variation (±3%)
        buy_ratio_variation = 1 + np.random.normal(0, 0.03, len(df))
        df['taker_buy_quote_asset_volume'] = df['quote_volume'] * original_buy_ratio * buy_ratio_variation
        # Ensure constraints
        df['taker_buy_quote_asset_volume'] = np.minimum(df['taker_buy_quote_asset_volume'], df['quote_volume'])
        df['taker_buy_quote_asset_volume'] = np.maximum(df['taker_buy_quote_asset_volume'], 0)
    
    if 'taker_buy_base_asset_volume' in df.columns and 'volume' in df.columns:
        # Calculate original taker buy ratio for base asset
        original_base_buy_ratio = df['taker_buy_base_asset_volume'] / (df['volume'] + 1e-8)
        # Apply ratio to new volume with small random variation (±3%)
        base_buy_ratio_variation = 1 + np.random.normal(0, 0.03, len(df))
        df['taker_buy_base_asset_volume'] = df['volume'] * original_base_buy_ratio * base_buy_ratio_variation
        # Ensure constraints
        df['taker_buy_base_asset_volume'] = np.minimum(df['taker_buy_base_asset_volume'], df['volume'])
        df['taker_buy_base_asset_volume'] = np.maximum(df['taker_buy_base_asset_volume'], 0)
    
    return df

# ===========================
# NOISE GENERATION FUNCTIONS
# ===========================

# Improved GBM with gravity (mean-reverting to original)
def apply_gbm_gravity_noise(df, symbol=None, **kwargs):
    """Simplified GBM with gravity using direct hourly parameters"""
    params = CUSTOM_PARAMETERS['GBM_Gravity']
    params.update(kwargs)
    
    df = df.copy()
    prices = ['open', 'close', 'high', 'low']
    
    # Calculate historical parameters directly from hourly returns
    if 'close' in df.columns:
        returns = df['close'].pct_change().dropna()
        if len(returns) > 10:
            # 直接使用小时收益率统计量，无需时间单位转换
            mu = returns.mean()  # 小时漂移率
            historical_vol = returns.std()  # 小时波动率
            sigma = historical_vol * params['sigma_scale']  # 缩放后的小时波动率
        else:
            mu, sigma = 0.0, 0.005  # Conservative defaults
    else:
        mu, sigma = 0.0, 0.005
    
    orig_prices = df[prices].copy()
    
    for col in prices:
        if col in df.columns:
            # 标准正态随机数，dt=1（1小时）时 dW ~ N(0,1)
            dW = np.random.normal(0, 1, len(df))
            for t in range(1, len(df)):
                if pd.notna(df[col].iloc[t-1]) and pd.notna(orig_prices[col].iloc[t]):
                    
                    drift = mu * df[col].iloc[t-1] * params['drift_scale']  # 漂移项
                    diffusion = sigma * df[col].iloc[t-1] * dW[t]  # 扩散项
                    # 引力项：拉向原始价格
                    gravity = params['G'] * (orig_prices[col].iloc[t] - df[col].iloc[t-1])
                    
                    new_price = df[col].iloc[t-1] + drift + diffusion + gravity
                    df.loc[df.index[t], col] = max(new_price, 1e-8)  # Ensure positive
    
    # Ensure OHLC constraints
    df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
    df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
    return adjust_linked_fields(df)

# GARCH-based noise with G parameter for mixing original and generated prices
def apply_garch_noise(df, symbol=None, **kwargs):
    params = CUSTOM_PARAMETERS['GARCH']
    params.update(kwargs)

    df = df.copy()
    G = params['G']  # Gravity parameter: 1.0=original GARCH, 0.0=full generation, 0.5=mixed

    if G >= 1.0:
        # Pure original GARCH behavior - modify existing prices
        return _apply_original_garch_logic(df, symbol, params)
    elif G <= 0.0:
        # Pure generation behavior
        return _apply_garch_generation_logic(df, symbol, params)
    else:
        # Mixed approach: blend original modification with generation
        # Apply original GARCH modification
        modified_df = _apply_original_garch_logic(df.copy(), symbol, params)

        # Apply GARCH generation
        generated_df = _apply_garch_generation_logic(df.copy(), symbol, params)

        # Blend the results based on G parameter
        prices = ['open', 'close', 'high', 'low']
        for col in prices:
            if col in df.columns:
                # G closer to 1: more weight to modified original prices
                # G closer to 0: more weight to generated prices
                df[col] = G * modified_df[col] + (1 - G) * generated_df[col]
                df[col] = np.maximum(df[col], 1e-8)

        # Ensure OHLC constraints
        df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
        df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))

        return adjust_linked_fields(df)

def _apply_original_garch_logic(df, symbol, params):
    """Original GARCH modification logic"""
    prices = ['open', 'close', 'high', 'low']

    for col in prices:
        if col in df.columns:
            returns = df[col].pct_change().dropna()
            if len(returns) > 20:  # Need more data for GARCH
                try:
                    model = arch_model(returns * 100, vol='Garch', p=1, q=1, rescale=False)
                    res = model.fit(disp='off', show_warning=False)
                    vol = res.conditional_volatility / 100  # Scale back

                    # Apply GARCH-based noise
                    noise = np.random.normal(0, 1, len(vol))
                    price_changes = vol * noise * params['sigma_scale']

                    # Apply to prices (skip first value due to pct_change)
                    for i in range(1, len(df)):
                        if i-1 < len(price_changes):
                            df.loc[df.index[i], col] *= (1 + price_changes.iloc[i-1])
                            df.loc[df.index[i], col] = max(df.loc[df.index[i], col], 1e-8)
                except:
                    # Fallback to simple noise if GARCH fails
                    symbol_info = f" for {symbol}" if symbol else ""
                    print(f"GARCH model failed for {col}{symbol_info}, using simple noise")
                    noise = np.random.normal(0, params['sigma_scale'] * 0.01, len(df))
                    df[col] *= (1 + noise)
                    df[col] = np.maximum(df[col], 1e-8)
            else:
                # Simple noise for insufficient data
                symbol_info = f" for {symbol}" if symbol else ""
                print(f"Not enough data for GARCH model{symbol_info}, using simple noise for {col}")
                noise = np.random.normal(0, params['sigma_scale'] * 0.01, len(df))
                df[col] *= (1 + noise)
                df[col] = np.maximum(df[col], 1e-8)

    # Ensure OHLC constraints
    df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
    df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
    return df

def _apply_garch_generation_logic(df, symbol, params):
    """GARCH generation logic (adapted from GARCH-Generate)"""
    # Work with close prices first (most important for GARCH modeling)
    if 'close' not in df.columns or len(df) < 50:
        symbol_info = f" for {symbol}" if symbol else ""
        print(f"Insufficient data for GARCH generation{symbol_info}, using original data")
        return df

    # Calculate returns from original close prices
    original_close = df['close'].dropna()
    if len(original_close) < 50:
        symbol_info = f" for {symbol}" if symbol else ""
        print(f"Not enough valid close prices for GARCH generation{symbol_info}, using original data")
        return df

    returns = original_close.pct_change().dropna()
    if len(returns) < 30:
        symbol_info = f" for {symbol}" if symbol else ""
        print(f"Not enough returns for GARCH generation{symbol_info}, using original data")
        return df

    try:
        # Fit GARCH model to original returns
        model = arch_model(returns * 100, vol='Garch', p=1, q=1, rescale=False)
        res = model.fit(disp='off', show_warning=False)

        # Extract model parameters
        omega = res.params['omega']
        alpha = res.params['alpha[1]']
        beta = res.params['beta[1]']

        # Calculate long-term volatility
        long_term_vol = np.sqrt(omega / (1 - alpha - beta)) / 100

        # Generate new return series using GARCH process
        new_returns = []
        vol_t = long_term_vol  # Initialize volatility
        drift = returns.mean() * params['drift_scale']

        for t in range(len(returns)):
            # GARCH volatility equation: σ²(t) = ω + α*ε²(t-1) + β*σ²(t-1)
            if t > 0:
                vol_t_squared = (omega/10000) + alpha * (new_returns[t-1]**2) + beta * (vol_t**2)
                vol_t = np.sqrt(max(vol_t_squared, 1e-8))

            # Scale volatility
            vol_t *= params['sigma_scale']

            # Generate new return
            epsilon = np.random.normal(0, 1)
            new_return = drift + vol_t * epsilon
            new_returns.append(new_return)

        new_returns = np.array(new_returns)

        # Generate new price series starting from original first price
        new_close_prices = [original_close.iloc[0]]
        for ret in new_returns:
            new_price = new_close_prices[-1] * (1 + ret)
            new_close_prices.append(max(new_price, 1e-8))

        new_close_prices = new_close_prices[1:]  # Remove the duplicate first price

        # Create new close price series aligned with original index
        close_index = original_close.index
        if len(new_close_prices) == len(close_index):
            df.loc[close_index, 'close'] = new_close_prices

        # Generate OHLC based on new close prices with realistic relationships
        prices = ['open', 'high', 'low']
        for col in prices:
            if col in df.columns:
                if col == 'open':
                    # Open prices: use previous close with some noise
                    df.loc[df.index[1:], 'open'] = df['close'].shift(1).iloc[1:] * (1 + np.random.normal(0, 0.002, len(df) - 1))
                    df.loc[df.index[0], 'open'] = df.loc[df.index[0], 'close']  # First open = first close
                elif col == 'high':
                    # High prices: max of open/close plus some upward bias
                    base_high = np.maximum(df['open'], df['close'])
                    high_premium = np.random.exponential(0.005, len(df))  # Exponential for realistic high spikes
                    df[col] = base_high * (1 + high_premium)
                elif col == 'low':
                    # Low prices: min of open/close minus some downward bias
                    base_low = np.minimum(df['open'], df['close'])
                    low_discount = np.random.exponential(0.005, len(df))  # Exponential for realistic low dips
                    df[col] = base_low * (1 - low_discount)

                # Ensure positive prices
                df[col] = np.maximum(df[col], 1e-8)

        # Final OHLC constraint enforcement
        df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
        df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))

    except Exception as e:
        symbol_info = f" for {symbol}" if symbol else ""
        print(f"GARCH generation failed{symbol_info}: {str(e)}, using original data")
        return df

    return df


# Improved anomaly injection with VaR-based scaling and gradual recovery
def apply_anomaly_injection_noise(df, symbol_var_dict, symbol, **kwargs):
    """Enhanced anomaly injection with VaR-based scaling and gradual recovery"""
    params = CUSTOM_PARAMETERS['Anomaly_Injection']
    params.update(kwargs)
    
    df = df.copy()
    prices = ['open', 'close', 'high', 'low']
    
    if len(df) < params['recovery_hours']:  # Not enough data for recovery mechanism
        return df
    
    # Get VaR for this symbol
    var_95 = symbol_var_dict.get(symbol, 0.02)  # Default 2% if not found
    
    # Determine anomaly events (sparse)
    num_anomalies = max(1, int(len(df) * params['anomaly_prob']))
    anomaly_indices = np.random.choice(
        range(params['recovery_hours'], len(df) - params['recovery_hours']), 
        size=min(num_anomalies, len(df) - 2 * params['recovery_hours']),
        replace=False
    )
    
    for anomaly_idx in anomaly_indices:
        # Dynamic jump scale based on VaR
        jump_multiplier = np.random.uniform(1.0, 3.0)  # 1-3x VaR
        jump_scale = var_95 * jump_multiplier
        jump_direction = np.random.choice([-1, 1])  # Up or down
        jump = jump_direction * jump_scale
        
        # Apply initial jump
        for col in prices:
            if col in df.columns:
                original_price = df.loc[df.index[anomaly_idx], col]
                df.loc[df.index[anomaly_idx], col] *= (1 + jump)
                df.loc[df.index[anomaly_idx], col] = max(df.loc[df.index[anomaly_idx], col], 1e-8)
        
        # Gradual recovery mechanism (exponential decay back to original)
        recovery_lambda = 3.0 / params['recovery_hours']  # Half-life parameter
        
        for h in range(1, params['recovery_hours']):
            if anomaly_idx + h < len(df):
                # Exponential decay factor
                decay_factor = np.exp(-recovery_lambda * h)
                recovery_adjustment = jump * decay_factor
                
                for col in prices:
                    if col in df.columns:
                        current_price = df.loc[df.index[anomaly_idx + h], col]
                        # Apply recovery adjustment
                        df.loc[df.index[anomaly_idx + h], col] *= (1 + recovery_adjustment)
                        df.loc[df.index[anomaly_idx + h], col] = max(df.loc[df.index[anomaly_idx + h], col], 1e-8)
        
        # Boost volume during anomaly period
        volume_boost_hours = min(6, params['recovery_hours'] // 6)  # Boost for first few hours
        volume_multiplier = 1 + abs(jump) * 5  # Higher activity during anomaly
        
        for h in range(volume_boost_hours):
            if anomaly_idx + h < len(df):
                volume_fields = ['volume', 'trade_num', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
                for field in volume_fields:
                    if field in df.columns:
                        boost_factor = volume_multiplier * np.exp(-0.3 * h)  # Decay over time
                        df.loc[df.index[anomaly_idx + h], field] *= boost_factor
    
    # Ensure OHLC constraints
    df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
    df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
    return adjust_linked_fields(df)

# Fixed pivot generation to match original time range
def generate_pivot_aligned(data_dict, fields, original_pivot=None):
    """Generate pivot aligned with original time range"""
    pivot = {}
    
    for field in fields:
        df_list = []
        for symbol, df in data_dict.items():
            # Handle different field names mapping
            source_field = field
            if field == 'vwap1m':
                # Use avg_price_1m as source for vwap1m
                if 'avg_price_1m' in df.columns:
                    source_field = 'avg_price_1m'
                else:
                    continue  # Skip if no suitable field
            elif field == 'funding_rate':
                # For swap data, use funding_fee if funding_rate not available
                if 'funding_rate' in df.columns:
                    source_field = 'funding_rate'
                elif 'funding_fee' in df.columns:
                    source_field = 'funding_fee'
                else:
                    continue
            
            if source_field in df.columns:
                s = df.set_index('candle_begin_time')[source_field].rename(symbol)
                df_list.append(s)
        
        if df_list:
            combined_pivot = pd.concat(df_list, axis=1)
            
            # Align with original pivot time range if provided
            if original_pivot is not None and field in original_pivot:
                orig_index = original_pivot[field].index
                # Reindex to match original time range exactly
                pivot[field] = combined_pivot.reindex(orig_index)
            else:
                pivot[field] = combined_pivot
    
    return pivot

# ===========================
# PARALLEL PROCESSING
# ===========================

# Worker function for parallel processing
def process_symbol_data(args):
    """Process a single symbol's data with noise function"""
    symbol, data, noise_func, data_type, extra_params = args
    try:
        if 'var_dict' in extra_params:
            # For anomaly injection, pass VaR dictionary and symbol
            processed_data = noise_func(data.copy(), extra_params['var_dict'], symbol)
        else:
            # Pass symbol to all noise functions for better error logging
            processed_data = noise_func(data.copy(), symbol=symbol)
        return symbol, processed_data, None
    except Exception as e:
        return symbol, None, str(e)

# Parallel processing function
def apply_noise_parallel(data_dict, noise_func, data_type, extra_params=None, max_workers=None):
    """Apply noise function to dictionary data in parallel"""
    if max_workers is None:
        max_workers = MAX_WORKERS
    
    if extra_params is None:
        extra_params = {}
    
    # Prepare arguments for parallel processing
    args_list = [(symbol, data, noise_func, data_type, extra_params) for symbol, data in data_dict.items()]
    
    processed_dict = {}
    failed_symbols = []
    
    print(f"Processing {len(args_list)} {data_type} symbols using {max_workers} workers...")
    
    with Pool(max_workers) as pool:
        # Use tqdm for progress bar
        results = list(tqdm(
            pool.imap(process_symbol_data, args_list),
            total=len(args_list),
            desc=f"Processing {data_type}",
            unit="symbols"
        ))
    
    # Collect results
    for symbol, processed_data, error in results:
        if error is None:
            processed_dict[symbol] = processed_data
        else:
            failed_symbols.append((symbol, error))
    
    if failed_symbols:
        print(f"Warning: Failed to process {len(failed_symbols)} {data_type} symbols:")
        for symbol, error in failed_symbols[:5]:  # Show first 5 errors
            print(f"  {symbol}: {error}")
        if len(failed_symbols) > 5:
            print(f"  ... and {len(failed_symbols) - 5} more")
    
    return processed_dict

# ===========================
# MAIN GENERATION FUNCTION
# ===========================

# Main function to generate parallel data
def generate_parallel_world(mode, output_dir):
    start_time = time.time()
    print(f"Generating parallel world with {mode} noise...")
    
    # Load data
    print("Loading data...")
    spot_dict, swap_dict = load_data()
    
    # Load original pivots for alignment
    print("Loading original pivot data for alignment...")
    try:
        with open(os.path.join(raw_data_path, 'market_pivot_spot.pkl'), 'rb') as f:
            orig_spot_pivot = pickle.load(f)
        with open(os.path.join(raw_data_path, 'market_pivot_swap.pkl'), 'rb') as f:
            orig_swap_pivot = pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load original pivots: {e}")
        orig_spot_pivot = orig_swap_pivot = None
    
    # Calculate VaR for anomaly injection
    var_dict = {}
    if mode == 'Anomaly_Injection':
        print("Calculating VaR for all symbols...")
        spot_var = calculate_var_dict(spot_dict)
        swap_var = calculate_var_dict(swap_dict)
        var_dict = {**spot_var, **swap_var}
        print(f"Calculated VaR for {len(var_dict)} symbols")
    
    # Noise function mapping
    noise_functions = {
        'GBM_Gravity': apply_gbm_gravity_noise,
        'GARCH': apply_garch_noise,
        'Anomaly_Injection': apply_anomaly_injection_noise
    }
    
    if mode not in noise_functions:
        raise ValueError(f"Unknown mode: {mode}")
    
    noise_func = noise_functions[mode]
    extra_params = {'var_dict': var_dict} if mode == 'Anomaly_Injection' else {}
    
    # Apply noise in parallel
    print(f"\nApplying {mode} noise...")
    spot_dict_modified = apply_noise_parallel(spot_dict, noise_func, "spot", extra_params)
    swap_dict_modified = apply_noise_parallel(swap_dict, noise_func, "swap", extra_params)
    
    # Generate pivots with alignment
    print("\nGenerating pivot tables...")
    with tqdm(total=2, desc="Creating pivots") as pbar:
        spot_pivot = generate_pivot_aligned(
            spot_dict_modified, 
            ['open', 'close', 'vwap1m'], 
            orig_spot_pivot
        )
        pbar.update(1)
        swap_pivot = generate_pivot_aligned(
            swap_dict_modified, 
            ['open', 'close', 'funding_rate', 'vwap1m'], 
            orig_swap_pivot
        )
        pbar.update(1)
    
    # Save files
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving to {output_dir}...")
    
    files_to_save = [
        ('market_pivot_spot.pkl', spot_pivot),
        ('market_pivot_swap.pkl', swap_pivot),
        ('spot_dict.pkl', spot_dict_modified),
        ('swap_dict.pkl', swap_dict_modified)
    ]
    
    with tqdm(total=len(files_to_save), desc="Saving files") as pbar:
        for filename, data in files_to_save:
            with open(os.path.join(output_dir, filename), 'wb') as f:
                pickle.dump(data, f)
            pbar.update(1)
    
    elapsed_time = time.time() - start_time
    print(f"\n✓ Successfully generated {mode} parallel world in {output_dir}")
    print(f"  Total time: {elapsed_time:.1f} seconds")
    print(f"  Spot symbols processed: {len(spot_dict_modified)}/{len(spot_dict)}")
    print(f"  Swap symbols processed: {len(swap_dict_modified)}/{len(swap_dict)}")
    
    # Verify pivot alignment
    if orig_spot_pivot is not None:
        for field in ['open', 'close', 'vwap1m']:
            if field in orig_spot_pivot and field in spot_pivot:
                orig_shape = orig_spot_pivot[field].shape
                new_shape = spot_pivot[field].shape
                if orig_shape == new_shape:
                    print(f"  ✓ Spot {field} pivot aligned: {new_shape}")
                else:
                    print(f"  ⚠️  Spot {field} pivot mismatch: {orig_shape} -> {new_shape}")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("PARALLEL HISTORY DATA GENERATION")
    print("=" * 60)
    print(f"Output directory: {processed_data_path}")
    print(f"Available CPU cores: {cpu_count()}")
    print(f"Using workers: {MAX_WORKERS}")
    print(f"Generation modes: {', '.join(GENERATION_MODES)}")
    print(f"Worlds per mode: {GENERATION_COUNT}")
    print(f"Random seed: {RANDOM_SEED} {'(reproducible)' if RANDOM_SEED is not None else '(different each run)'}")
    print()
    
    print("CUSTOM PARAMETERS:")
    for mode, params in CUSTOM_PARAMETERS.items():
        print(f"  {mode}: {params}")
    print()
    
    # Set global random seed if specified
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)
        print(f"✓ Global random seed set to: {RANDOM_SEED}")
        print()
    
    overall_start_time = time.time()
    successful_generations = []
    failed_generations = []
    
    # Generate multiple worlds for each mode
    world_count = 1
    for mode in GENERATION_MODES:
        for i in range(GENERATION_COUNT):
            output_dir = os.path.join(processed_data_path, f'{mode}_{i+1}')
            print(f"[{world_count}/{len(GENERATION_MODES) * GENERATION_COUNT}] Starting {mode} generation (World {i+1})...")
            
            try:
                success = generate_parallel_world(mode, output_dir)
                if success:
                    successful_generations.append(f"{mode}_{i+1}")
                    print(f"✓ {mode} world {i+1} completed successfully\n")
                else:
                    failed_generations.append(f"{mode}_{i+1}")
                    print(f"✗ {mode} world {i+1} failed\n")
            except Exception as e:
                failed_generations.append(f"{mode}_{i+1}")
                print(f"✗ Failed to generate {mode} world {i+1}: {e}\n")
            
            world_count += 1
    
    # Final summary
    total_time = time.time() - overall_start_time
    total_worlds = len(GENERATION_MODES) * GENERATION_COUNT
    print("=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Successful: {len(successful_generations)}/{total_worlds}")
    if successful_generations:
        print(f"  ✓ {', '.join(successful_generations)}")
    if failed_generations:
        print(f"Failed: {len(failed_generations)}/{total_worlds}")
        print(f"  ✗ {', '.join(failed_generations)}")
    print()
    
    if successful_generations:
        print("🎉 Parallel history data generation completed successfully!")
    else:
        print("✗ All generations failed!")
    print("=" * 60) 