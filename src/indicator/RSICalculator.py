import numpy as np
import pandas as pd


class RSICalculator:
    """RSI指标计算和分析类"""

    @staticmethod
    def calculate_traditional_rsi(prices, period=14, smoothing_type='simple'):
        """
        计算传统RSI指标

        参数:
        prices (pd.Series): 价格序列
        period (int): RSI计算周期，默认14
        smoothing_type (str): 平滑方法，可选'simple'（简单平均）或'exponential'（指数平滑）

        返回:
        pd.Series: RSI值序列
        """
        if len(prices) < period + 1:
            return pd.Series(np.nan, index=prices.index)

        # 计算价格变化
        deltas = prices.diff().dropna()

        # 分离上涨和下跌
        gains = deltas.clip(lower=0)
        losses = -deltas.clip(upper=0)

        # 根据平滑方法计算平均涨跌幅
        if smoothing_type == 'simple':
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
        elif smoothing_type == 'exponential':
            avg_gains = gains.ewm(com=period - 1, min_periods=period).mean()
            avg_losses = losses.ewm(com=period - 1, min_periods=period).mean()
        else:
            raise ValueError(f"不支持的平滑方法: {smoothing_type}")

        # 避免除零错误
        avg_losses = avg_losses.replace(0, 1e-10)

        # 计算相对强度和RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_cutler_rsi(prices, period=14):
        """
        计算Cutler's RSI，这是一个不同于标准RSI的变种

        参数:
        prices (pd.Series): 价格序列
        period (int): 计算周期

        返回:
        pd.Series: Cutler's RSI值序列
        """
        # 使用Rolling百分比变化而非价格差异
        up_days = prices.rolling(period).apply(lambda x: sum(1 for i in range(1, len(x)) if x[i] > x[i - 1]), raw=True)
        rsi = 100 * up_days / period
        return rsi

    @staticmethod
    def calculate_stochastic_rsi(prices, rsi_period=14, stoch_period=14):
        """
        计算随机RSI (StochRSI)

        参数:
        prices (pd.Series): 价格序列
        rsi_period (int): RSI计算周期
        stoch_period (int): StochRSI周期

        返回:
        pd.Series: 随机RSI值序列
        """
        # 计算传统RSI
        rsi = RSICalculator.calculate_traditional_rsi(prices, rsi_period)

        # 计算RSI的最高和最低值
        rsi_min = rsi.rolling(window=stoch_period).min()
        rsi_max = rsi.rolling(window=stoch_period).max()

        # 计算随机RSI
        stoch_rsi = 100 * (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)

        return stoch_rsi

    @staticmethod
    def detect_rsi_divergence(prices, rsi, window=10):
        """
        检测RSI背离 - 改进版
        增加了数据验证和异常处理，防止小样本统计问题

        参数:
        prices (pd.Series): 价格序列
        rsi (pd.Series): RSI序列
        window (int): 检测窗口大小

        返回:
        dict: 包含背离类型和强度
        """
        # 默认返回值
        default_result = {"bullish": False, "bearish": False, "strength": 0}

        # 数据验证
        if prices is None or rsi is None:
            return default_result

        # 确保数据长度足够
        if len(prices) < max(window, 5) or len(rsi) < max(window, 5):
            return default_result

        try:
            # 获取最近的数据
            recent_prices = prices[-window:].reset_index(drop=True)
            recent_rsi = rsi[-window:].reset_index(drop=True)

            # 确保没有缺失值
            if recent_prices.isna().any() or recent_rsi.isna().any():
                # 清理缺失值
                valid_indices = ~(recent_prices.isna() | recent_rsi.isna())
                recent_prices = recent_prices[valid_indices]
                recent_rsi = recent_rsi[valid_indices]

                # 再次检查数据长度
                if len(recent_prices) < 5 or len(recent_rsi) < 5:
                    return default_result

            # 计算数据点的简单趋势（采用线性回归而非简单比较）
            try:
                # 使用线性回归来确定趋势
                x = np.arange(len(recent_prices))
                price_slope = np.polyfit(x, recent_prices, 1)[0]
                rsi_slope = np.polyfit(x, recent_rsi, 1)[0]

                price_trend = price_slope > 0
                rsi_trend = rsi_slope > 0
            except Exception as trend_e:
                # 回退到简单比较
                price_trend = recent_prices.iloc[-1] > recent_prices.iloc[0]
                rsi_trend = recent_rsi.iloc[-1] > recent_rsi.iloc[0]

            # 使用更安全的相关性计算，避免除零错误
            try:
                # 确保有足够的变异用于计算相关性
                if recent_prices.std() > 1e-10 and recent_rsi.std() > 1e-10:
                    correlation = np.corrcoef(recent_prices, recent_rsi)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0
                else:
                    correlation = 0
            except Exception as corr_e:
                # 如果相关性计算失败，使用默认值
                correlation = 0

            # 计算背离强度
            strength = abs(correlation)

            # 判断背离类型
            bullish_divergence = not price_trend and rsi_trend  # 价格下降但RSI上升
            bearish_divergence = price_trend and not rsi_trend  # 价格上升但RSI下降

            # 返回结果
            return {
                "bullish": bullish_divergence,
                "bearish": bearish_divergence,
                "strength": strength
            }
        except Exception as e:
            print(f"计算RSI背离时出错: {e}")
            return default_result

    @staticmethod
    def analyze_rsi_conditions(rsi, overbought=70, oversold=30, neutral_zone=None):
        """
        分析RSI的超买超卖状态

        参数:
        rsi (pd.Series): RSI序列
        overbought (float): 超买阈值
        oversold (float): 超卖阈值
        neutral_zone (tuple): 中性区间，如(40, 60)

        返回:
        dict: RSI分析结果
        """
        if neutral_zone is None:
            neutral_zone = (oversold + 10, overbought - 10)

        latest_rsi = rsi.iloc[-1]

        # 基本状态判断
        is_overbought = latest_rsi >= overbought
        is_oversold = latest_rsi <= oversold
        is_neutral = neutral_zone[0] <= latest_rsi <= neutral_zone[1]

        # RSI趋势分析（最近3天）
        if len(rsi) >= 3:
            recent_trend = "上升" if rsi.iloc[-1] > rsi.iloc[-3] else "下降" if rsi.iloc[-1] < rsi.iloc[-3] else "盘整"
        else:
            recent_trend = "数据不足"

        # 检测超买超卖区域的反转信号
        reversal_signal = False
        if len(rsi) >= 3:
            if is_overbought and rsi.iloc[-1] < rsi.iloc[-2]:
                reversal_signal = "顶部反转可能"
            elif is_oversold and rsi.iloc[-1] > rsi.iloc[-2]:
                reversal_signal = "底部反转可能"

        return {
            "current_value": latest_rsi,
            "status": "超买" if is_overbought else "超卖" if is_oversold else "中性",
            "trend": recent_trend,
            "reversal_signal": reversal_signal if reversal_signal else "无明显反转信号",
            "is_actionable": is_overbought or is_oversold  # 是否需要采取行动
        }