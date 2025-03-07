import akshare as ak
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import traceback


def get_stock_data(code, name, period="daily", start_date="20220101", end_date="20250101"):
    """获取单个股票的历史数据"""
    try:
        # 尝试不同的股票代码格式
        # AKShare可能需要的是直接使用原始代码而不添加前缀
        try:
            stock_data = ak.stock_zh_a_hist(symbol=code, period=period,
                                            start_date=start_date, end_date=end_date,
                                            adjust="qfq")
        except Exception as e1:
            # print(f"所有代码格式均失败，放弃获取 {code} {name}, exception: {e1}")
            # return None
            print(f"尝试直接使用代码 {code} 失败: {e1}")

            # 尝试添加前缀
            if code.startswith('6'):
                full_code = f"sh{code}"
            else:
                full_code = f"sz{code}"

            try:
                stock_data = ak.stock_zh_a_hist(symbol=full_code, period=period,
                                                start_date=start_date, end_date=end_date,
                                                adjust="qfq")
            except Exception as e2:
                print(f"尝试使用带前缀的代码 {full_code} 失败: {e2}")

                # 最后尝试使用不带前缀但不同格式的代码
                try:
                    # 东方财富格式
                    if code.startswith('6'):
                        df_code = f"1.{code}"
                    else:
                        df_code = f"0.{code}"

                    stock_data = ak.stock_zh_a_hist(symbol=df_code, period=period,
                                                    start_date=start_date, end_date=end_date,
                                                    adjust="qfq")
                except Exception as e3:
                    print(f"所有代码格式均失败，放弃获取 {code}-{name}, 失败: {e3}")
                    return None

        # 检查数据是否足够
        if len(stock_data) < 30:
            print(f"股票 {code} {name} 数据不足30条，跳过")
            return None

        # 检查接口返回的列名
        print(f"股票 {code} 返回的列名: {stock_data.columns.tolist()}")

        # 设置日期为索引 (根据实际列名调整)
        date_column = '日期' if '日期' in stock_data.columns else 'date'
        stock_data[date_column] = pd.to_datetime(stock_data[date_column])
        stock_data.set_index(date_column, inplace=True)

        return stock_data
    except Exception as e:
        print(f"获取 {code} {name} 数据时出错: {e}")
        print(traceback.format_exc())
        return None


def calculate_indicators(data):
    """计算技术指标"""
    if data is None or len(data) < 30:
        return None

    try:
        # 打印列名，便于调试
        print(f"计算指标前的列名: {data.columns.tolist()}")

        # 获取价格列的名称
        # 根据AKShare返回的实际列名进行调整
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

        # 计算RSI
        def calculate_rsi(series, periods=14):
            delta = series.diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            avg_gain = up.rolling(window=periods).mean()
            avg_loss = down.rolling(window=periods).mean()
            # 避免除零错误
            avg_loss = avg_loss.replace(0, 0.001)
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        df['rsi6'] = calculate_rsi(df[close_col], periods=6)
        df['rsi12'] = calculate_rsi(df[close_col], periods=12)
        df['rsi14'] = calculate_rsi(df[close_col], periods=14)
        df['rsi24'] = calculate_rsi(df[close_col], periods=24)

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

        return df

    except Exception as e:
        print(f"计算指标时出错: {e}")
        print(traceback.format_exc())
        return None


class AStockScreener:
    def __init__(self):
        self.all_stocks = None
        self.stock_data = {}
        self.filtered_stocks = []

    def get_all_stocks(self):
        """获取所有A股的股票代码和名称"""
        print("正在获取所有A股股票列表...")
        try:
            stock_info = ak.stock_info_a_code_name()
            # 只保留主板、中小板和创业板，去掉北交所等
            stock_info = stock_info[stock_info['code'].apply(lambda x: x.startswith(('60', '00', '30')))]
            self.all_stocks = stock_info
            print(f"共获取到 {len(self.all_stocks)} 只股票")
            return self.all_stocks
        except Exception as e:
            print(f"获取A股列表出错: {e}")
            print(traceback.format_exc())
            return pd.DataFrame()

    # 其余方法保持不变，但在内部使用时需要注意列名的匹配
    def process_stock(self, code, name):
        """处理单个股票的完整流程"""
        # 获取股票数据
        data = get_stock_data(code, name)
        if data is None:
            return None

        # 计算技术指标
        data_with_indicators = calculate_indicators(data)
        if data_with_indicators is None:
            return None

        # 保存处理后的数据
        self.stock_data[code] = data_with_indicators
        return data_with_indicators

    def parallel_process_stocks(self, max_workers=5, max_stocks=None):
        """并行处理多只股票"""
        if self.all_stocks is None:
            self.get_all_stocks()

        # 可以限制处理的股票数量，用于测试
        stocks_to_process = self.all_stocks
        if max_stocks:
            stocks_to_process = self.all_stocks.head(max_stocks)

        print(f"开始处理 {len(stocks_to_process)} 只股票的数据...")

        # 降低并发数，避免过快请求导致被限制
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 创建任务列表
            future_to_stock = {
                executor.submit(self.process_stock, row['code'], row['name']):
                    row['code'] for _, row in stocks_to_process.iterrows()
            }

            # 使用tqdm显示进度条
            for future in tqdm(as_completed(future_to_stock), total=len(future_to_stock)):
                stock_code = future_to_stock[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"处理股票 {stock_code} 时发生错误: {e}")

        print(f"成功处理 {len(self.stock_data)} 只股票的数据")
        return self.stock_data

    def screen_kdj_golden_cross(self, days_window=5):
        """改进的KDJ金叉策略筛选
        寻找最近days_window天内发生的KDJ金叉
        """
        filtered = []
        for code, data in self.stock_data.items():
            # 检查数据是否足够
            if data is None or len(data) <= days_window + 2:
                continue

            try:
                # 获取最近一段时间的数据
                recent_data = data.iloc[-days_window - 2:]

                # 确保有足够的数据行
                if len(recent_data) < 3:
                    continue

                # 寻找窗口期内的KDJ金叉
                for i in range(1, min(days_window, len(recent_data) - 1)):
                    if (recent_data['k'].iloc[i - 1] < recent_data['d'].iloc[i - 1] and
                            recent_data['k'].iloc[i] > recent_data['d'].iloc[i]):
                        # 发现金叉
                        filtered.append(code)
                        break

            except Exception as e:
                print(f"处理股票 {code} 的KDJ数据时出错: {e}")
                continue

        return filtered

    def screen_macd_golden_cross(self, days_window=5):
        """改进的MACD金叉策略筛选
        寻找最近days_window天内发生的MACD金叉
        """
        filtered = []
        for code, data in self.stock_data.items():
            # 检查数据是否足够
            if data is None or len(data) <= days_window + 2:
                continue

            try:
                # 获取最近一段时间的数据
                recent_data = data.iloc[-days_window - 2:]

                # 确保有足够的数据行
                if len(recent_data) < 3:
                    continue

                # 寻找窗口期内的MACD金叉
                for i in range(1, min(days_window, len(recent_data) - 1)):
                    if (recent_data['macd'].iloc[i - 1] < recent_data['macd_signal'].iloc[i - 1] and
                            recent_data['macd'].iloc[i] > recent_data['macd_signal'].iloc[i]):
                        # 发现金叉
                        filtered.append(code)
                        break

            except Exception as e:
                print(f"处理股票 {code} 的MACD数据时出错: {e}")
                continue

        return filtered

    def screen_rsi_oversold_bounce(self, days_window=5, rsi_threshold=35):
        """改进的RSI超卖反弹策略筛选
        寻找最近days_window天内RSI从超卖区反弹的情况
        提高RSI阈值，放宽条件
        """
        filtered = []
        for code, data in self.stock_data.items():
            # 检查数据是否足够
            if data is None or len(data) <= days_window + 2:
                continue

            try:
                # 获取最近一段时间的数据
                recent_data = data.iloc[-days_window - 2:]

                # 确保有足够的数据行
                if len(recent_data) < 3:
                    continue

                # 寻找窗口期内的RSI反弹
                for i in range(1, min(days_window, len(recent_data) - 1)):
                    # 条件1: RSI低于阈值
                    # 条件2: RSI开始上升
                    if (recent_data['rsi14'].iloc[i - 1] < rsi_threshold and
                            recent_data['rsi14'].iloc[i - 1] < recent_data['rsi14'].iloc[i] < recent_data['rsi14'].iloc[
                                i + 1]):
                        # 发现RSI反弹
                        filtered.append(code)
                        break

            except Exception as e:
                print(f"处理股票 {code} 的RSI数据时出错: {e}")
                continue

        return filtered

    def screen_combined_strategy(self, days_window=5):
        """改进的组合策略筛选
        只要满足任意一种信号即可入选
        """
        # 获取各个单独策略的结果
        kdj_stocks = set(self.screen_kdj_golden_cross(days_window))
        macd_stocks = set(self.screen_macd_golden_cross(days_window))
        rsi_stocks = set(self.screen_rsi_oversold_bounce(days_window))

        # 使用"或"的逻辑，只要符合任一条件即可
        all_filtered = list(kdj_stocks | macd_stocks | rsi_stocks)

        # 增加一个权重排序逻辑
        weighted_stocks = []
        for code in all_filtered:
            weight = 0
            if code in kdj_stocks:
                weight += 1
            if code in macd_stocks:
                weight += 1
            if code in rsi_stocks:
                weight += 1
            weighted_stocks.append((code, weight))

        # 按权重排序
        weighted_stocks.sort(key=lambda x: x[1], reverse=True)

        # 返回排序后的股票代码
        return [code for code, _ in weighted_stocks]

    def run_screening(self, max_stocks=None, strategy='macd', days_ago=5):
        """运行完整的筛选流程"""
        # 获取所有股票
        if self.all_stocks is None:
            self.get_all_stocks()
        print(f"strategy={strategy}")
        # 处理股票数据
        self.parallel_process_stocks(max_stocks=max_stocks)

        # 根据选择的策略筛选股票
        if strategy == 'kdj':
            self.filtered_stocks = self.screen_kdj_golden_cross(days_ago)
        elif strategy == 'macd':
            self.filtered_stocks = self.screen_macd_golden_cross(days_ago)
        elif strategy == 'rsi':
            self.filtered_stocks = self.screen_rsi_oversold_bounce(days_ago)
        # elif strategy == 'boll':
        #     self.filtered_stocks = self.screen_boll_breakout(days_ago)
        # elif strategy == 'ema':
        #     self.filtered_stocks = self.screen_ema_crossover(days_ago)
        elif strategy == 'combined':
            self.filtered_stocks = self.screen_combined_strategy(days_ago)
        else:
            raise ValueError(f"不支持的策略: {strategy}")

        # 获取筛选出的股票的名称
        filtered_with_names = []
        for code in self.filtered_stocks:
            name = self.all_stocks[self.all_stocks['code'] == code]['name'].iloc[0]
            filtered_with_names.append((code, name))

        return filtered_with_names