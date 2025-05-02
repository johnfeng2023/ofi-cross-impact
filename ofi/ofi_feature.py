import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

### -----------------------------
### 1. Best-Level OFI
### -----------------------------
def compute_best_level_ofi(df):
    df = df.copy()
    df['bid_px_00_prev'] = df['bid_px_00'].shift(1)
    df['bid_sz_00_prev'] = df['bid_sz_00'].shift(1)
    df['ask_px_00_prev'] = df['ask_px_00'].shift(1)
    df['ask_sz_00_prev'] = df['ask_sz_00'].shift(1)

    def calc_bid(row):
        if row['bid_px_00'] > row['bid_px_00_prev']:
            return row['bid_sz_00']
        elif row['bid_px_00'] == row['bid_px_00_prev']:
            return row['bid_sz_00'] - row['bid_sz_00_prev']
        else:
            return -row['bid_sz_00']

    def calc_ask(row):
        if row['ask_px_00'] > row['ask_px_00_prev']:
            return -row['ask_sz_00']
        elif row['ask_px_00'] == row['ask_px_00_prev']:
            return row['ask_sz_00_prev'] - row['ask_sz_00']
        else:
            return row['ask_sz_00']

    df['Best_level_OFI'] = df.apply(calc_bid, axis=1) - df.apply(calc_ask, axis=1)
    df['minute'] = df['ts_event'].dt.floor('min')
    return df.groupby(['symbol', 'minute'])['Best_level_OFI'].sum().reset_index()


### -----------------------------
### 2. Multi-Level OFI
### -----------------------------
def compute_multi_level_ofi(df):
    df = df.copy()
    for level in range(10):
        for side in ['bid', 'ask']:
            for typ in ['px', 'sz']:
                col = f"{side}_{typ}_0{level}"
                df[f"{col}_prev"] = df[col].shift(1)

    def ofi_component(df, level, side):
        px, sz = f'{side}_px_0{level}', f'{side}_sz_0{level}'
        px_prev, sz_prev = f'{px}_prev', f'{sz}_prev'
        if side == 'bid':
            return np.where(df[px] > df[px_prev], df[sz],
                            np.where(df[px] == df[px_prev], df[sz] - df[sz_prev], -df[sz]))
        else:
            return np.where(df[px] > df[px_prev], -df[sz],
                            np.where(df[px] == df[px_prev], df[sz_prev] - df[sz], df[sz]))

    df['Multi_level_OFI'] = 0
    for level in range(10):
        bid_ofi = ofi_component(df, level, 'bid')
        ask_ofi = ofi_component(df, level, 'ask')
        df[f'ofi_level_{level}'] = bid_ofi - ask_ofi
        df['Multi_level_OFI'] += df[f'ofi_level_{level}']

    df['minute'] = df['ts_event'].dt.floor('min')
    return df, df.groupby(['symbol', 'minute'])['Multi_level_OFI'].sum().reset_index()


### -----------------------------
### 3. Integrated OFI (PCA-based)
### -----------------------------
def compute_integrated_ofi(df):
    df = df.copy()
    ofi_levels = [f'ofi_level_{i}' for i in range(10)]
    ofi_matrix = df.groupby(['symbol', df['ts_event'].dt.floor('min')])[ofi_levels].sum()

    # Compute average depth (Q)
    depth_cols = [f'bid_sz_0{i}' for i in range(10)] + [f'ask_sz_0{i}' for i in range(10)]
    df['minute'] = df['ts_event'].dt.floor('min')
    avg_depth = df.groupby(['symbol', 'minute'])[depth_cols].mean()
    avg_depth['Q'] = avg_depth.sum(axis=1) / 10

    ofi_matrix['Q'] = avg_depth['Q']
    for i in range(10):
        ofi_matrix[f'norm_ofi_{i}'] = ofi_matrix[f'ofi_level_{i}'] / ofi_matrix['Q']

    X = ofi_matrix[[f'norm_ofi_{i}' for i in range(10)]].dropna().values
    pca = PCA(n_components=1)
    integrated = pca.fit_transform(X).flatten()
    weights = pca.components_[0]
    norm_integrated = integrated / np.sum(np.abs(weights))

    integrated_df = ofi_matrix.dropna().reset_index()
    integrated_df['Integrated_OFI'] = norm_integrated
    return integrated_df[['symbol', 'ts_event', 'Integrated_OFI']]


### -----------------------------
### 4. Cross-Asset OFI (Lasso Regression)
### -----------------------------

def simulate_cross_asset_ofi(integrated_df, num_assets=2, noise_levels=(0.5, 0.7), base_col='Integrated_OFI'):
    """
    Simulate synthetic OFI signals based on a base signal (e.g., AAPL Integrated_OFI),
    adding Gaussian noise to generate synthetic stocks.

    Args:
        integrated_df: DataFrame with Integrated_OFI.
        num_assets: Number of synthetic assets to generate.
        noise_levels: Tuple/list of std devs or single value.
        base_col: Name of the base signal column.

    Returns:
        DataFrame with synthetic stocks and return of base as target.
    """
    df = integrated_df.rename(columns={base_col: 'AAPL'}).copy()
    np.random.seed(42)

    if isinstance(noise_levels, (float, int)):
        noise_levels = [noise_levels] * num_assets
    elif len(noise_levels) < num_assets:
        noise_levels = list(noise_levels) + [noise_levels[-1]] * (num_assets - len(noise_levels))

    
    synthetic_cols = {}
    for i in range(num_assets):
        name = f'STOCK_{i:03d}'
        synthetic_cols[name] = df['AAPL'] + np.random.normal(0, noise_levels[i], size=len(df))

    # Add all columns at once to avoid fragmentation
    df = pd.concat([df, pd.DataFrame(synthetic_cols)], axis=1)


    df['return_AAPL'] = df['AAPL'].shift(-1) - df['AAPL']
    return df.dropna()


def cross_asset_lasso(df, target_col='return_AAPL'):
    """
    Runs Lasso regression to predict return using synthetic cross-asset OFIs.
    """
    # Keep only numeric features and exclude the target column
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_col]
    X = df[feature_cols].values
    y = df[target_col].values

    X_scaled = StandardScaler().fit_transform(X)
    lasso = LassoCV(cv=5, random_state=0).fit(X_scaled, y)

    coefs = dict(zip(feature_cols, lasso.coef_))
    r2 = lasso.score(X_scaled, y)
    return coefs, r2

