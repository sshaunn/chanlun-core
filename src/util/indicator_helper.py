import traceback

import pandas as pd

from src.indicator.RSICalculator import RSICalculator


def calculate_indicators(data, rsi_parameters=None):
    """
    计算技术指标，增强版本

    参数:
    data (pd.DataFrame): 股票历史数据
    rsi_parameters (dict): RSI计算参数
    """
    if data is None or len(data) < 30:
        return None

    # 默认RSI参数
    if rsi_parameters is None:
        rsi_parameters = {
            'periods': [6, 12, 14, 24],
            'smoothing_type': 'simple',
            'include_stochastic': True,
            'stoch_period': 14
        }

    try:
        # 识别列名
        print(f"计算指标前的列名: {data.columns.tolist()}")
        close_col = '收盘' if '收盘' in data.columns else 'close'
        open_col = '开盘' if '开盘' in data.columns else 'open'
        high_col = '最高' if '最高' in data.columns else 'high'
        low_col = '最低' if '最低' in data.columns else 'low'
        volume_col = '成交量' if '成交量' in data.columns else 'volume'

        # 创建数据副本
        df = data.copy()

        # 计算移动平均线
        df['ma5'] = df[close_col].rolling(window=5).mean()
        df['ma10'] = df[close_col].rolling(window=10).mean()
        df['ma20'] = df[close_col].rolling(window=20).mean()
        df['ma60'] = df[close_col].rolling(window=60).mean()

        # 计算指数移动平均线
        df['ema5'] = df[close_col].ewm(span=5, adjust=False).mean()
        df['ema10'] = df[close_col].ewm(span=10, adjust=False).mean()
        df['ema20'] = df[close_col].ewm(span=20, adjust=False).mean()
        df['ema60'] = df[close_col].ewm(span=60, adjust=False).mean()

        # 计算MACD
        df['ema12'] = df[close_col].ewm(span=12, adjust=False).mean()
        df['ema26'] = df[close_col].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # 使用增强的RSI计算器
        for period in rsi_parameters['periods']:
            period_key = f'rsi{period}'
            df[period_key] = RSICalculator.calculate_traditional_rsi(
                df[close_col],
                period=period,
                smoothing_type=rsi_parameters['smoothing_type']
            )

        # 如果需要，计算随机RSI
        if rsi_parameters.get('include_stochastic', False):
            # 使用14天作为默认RSI周期计算随机RSI
            df['stoch_rsi'] = RSICalculator.calculate_stochastic_rsi(
                df[close_col],
                rsi_period=14,
                stoch_period=rsi_parameters.get('stoch_period', 14)
            )

        # 计算BOLL指标 (布林带)
        def calculate_bollinger_bands(series, window=20, num_std=2):
            rolling_mean = series.rolling(window=window).mean()
            rolling_std = series.rolling(window=window).std()
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            return upper_band, rolling_mean, lower_band

        df['upper'], df['middle'], df['lower'] = calculate_bollinger_bands(df[close_col])

        # 计算KDJ指标
        def calculate_kdj(this_df, n=9, m1=3, m2=3):
            low_min = this_df[low_col].rolling(window=n).min()
            high_max = this_df[high_col].rolling(window=n).max()

            # 避免除零错误
            denom = high_max - low_min
            denom = denom.replace(0, 0.001)

            # 计算 RSV
            rsv = 100 * ((this_df[close_col] - low_min) / denom)

            # 计算K值
            k = pd.Series(0.0, index=this_df.index)
            d = pd.Series(0.0, index=this_df.index)
            j = pd.Series(0.0, index=this_df.index)

            for i in range(len(this_df)):
                if i == 0:
                    k.iloc[i] = 50
                    d.iloc[i] = 50
                else:
                    k.iloc[i] = (m1 - 1) * k.iloc[i - 1] / m1 + rsv.iloc[i] / m1
                    d.iloc[i] = (m2 - 1) * d.iloc[i - 1] / m2 + k.iloc[i] / m2
                j.iloc[i] = 3 * k.iloc[i] - 2 * d.iloc[i]

            return k, d, j

        df['k'], df['d'], df['j'] = calculate_kdj(df)

        # 添加RSI背离检测
        if len(df) > 20:  # 确保有足够的数据来检测背离
            df['rsi_divergence'] = pd.Series([
                RSICalculator.detect_rsi_divergence(
                    df[close_col].iloc[:i + 1],
                    df['rsi14'].iloc[:i + 1],
                    window=min(10, i + 1)
                )
                for i in range(len(df))
            ])

        return df

    except Exception as e:
        print(f"计算指标时出错: {e}")
        print(traceback.format_exc())
        return None
