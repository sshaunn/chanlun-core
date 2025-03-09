import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import lru_cache

import akshare as ak
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.util.indicator_helper import calculate_indicators


class AStockScreener:
    def __init__(self, cache_dir="./stock_cache"):
        self.all_stocks = None
        self.stock_data = {}
        self.filtered_stocks = []
        self.cache_dir = cache_dir

        # 创建缓存目录
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    @lru_cache(maxsize=1)
    def get_all_stocks(self):
        """获取所有A股的股票代码和名称，使用缓存减少API调用"""
        cache_file = os.path.join(self.cache_dir, "all_stocks.json")

        # 检查缓存是否存在且未过期（1天内）
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time < timedelta(days=1):
                print("使用缓存的股票列表...")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    stocks_dict = json.load(f)
                    self.all_stocks = pd.DataFrame(stocks_dict)
                    print(f"从缓存加载了 {len(self.all_stocks)} 只股票")
                    return self.all_stocks

        print("正在获取所有A股股票列表...")
        try:
            stock_info = ak.stock_info_a_code_name()
            # 只保留主板、中小板和创业板，去掉北交所等
            stock_info = stock_info[stock_info['code'].apply(lambda x: x.startswith(('60', '00', '30')))]
            self.all_stocks = stock_info

            # 保存到缓存
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(stock_info.to_dict('records'), f, ensure_ascii=False)

            print(f"共获取到 {len(self.all_stocks)} 只股票")
            return self.all_stocks
        except Exception as e:
            print(f"获取A股列表出错: {e}")
            print(traceback.format_exc())

            # 如果获取失败但缓存存在，尝试使用过期缓存
            if os.path.exists(cache_file):
                print("使用过期缓存...")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    stocks_dict = json.load(f)
                    self.all_stocks = pd.DataFrame(stocks_dict)
                    return self.all_stocks

            return pd.DataFrame()

    def get_stock_data(self, code, name, period="daily", start_date="20220101", end_date="20250101", use_cache=True):
        """
        获取单个股票的历史数据，支持缓存
        修复版本：避免使用有问题的stock_zh_a_daily函数
        """
        # 缓存文件名逻辑保持不变
        cache_file = os.path.join(self.cache_dir, f"{code}_data.json")

        # 从缓存加载数据的逻辑保持不变
        if use_cache and os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time < timedelta(days=1):
                print(f"使用缓存的股票数据: {code}")
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data_dict = json.load(f)
                        if data_dict:
                            stock_data = pd.DataFrame(data_dict)
                            date_columns = ['日期', 'date', '时间', 'trade_date', 'datetime', 'Date']

                            for col in date_columns:
                                if col in stock_data.columns:
                                    stock_data[col] = pd.to_datetime(stock_data[col])
                                    stock_data.set_index(col, inplace=True)
                                    if len(stock_data) >= 30:
                                        return stock_data
                                    break
                except Exception as e:
                    print(f"读取缓存出错: {e}")

        # 主要的数据获取逻辑
        try:
            # 定义可能的股票代码格式
            code_formats = []
            code_formats.append(code)  # 原始代码

            # 根据代码前缀添加格式化代码
            if code.startswith('6'):
                code_formats.extend([f"sh{code}", f"SH{code}", f"1.{code}"])
            else:
                code_formats.extend([f"sz{code}", f"SZ{code}", f"0.{code}"])

            code_formats.append(code.lstrip('0'))  # 去掉前导零

            # 尝试所有代码格式
            stock_data = None
            successful_format = None

            # 1. 首先尝试stock_zh_a_hist
            for code_format in code_formats:
                try:
                    stock_data = ak.stock_zh_a_hist(
                        symbol=code_format,
                        period=period,
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq"
                    )

                    if stock_data is not None and not stock_data.empty:
                        successful_format = f"stock_zh_a_hist with {code_format}"
                        break
                except Exception as e:
                    print(f"使用stock_zh_a_hist尝试代码 {code_format} 失败: {e}")
                    continue

            # 2. 如果以上方法都失败，尝试使用东方财富数据源
            if stock_data is None or stock_data.empty:
                try:
                    print(f"尝试使用东方财富数据源获取 {code} 数据")
                    stock_data = ak.stock_zh_a_hist_163(
                        symbol=code,
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq"
                    )

                    if stock_data is not None and not stock_data.empty:
                        successful_format = "stock_zh_a_hist_163"
                except Exception as e:
                    print(f"使用stock_zh_a_hist_163获取 {code} 失败: {e}")

            # 3. 尝试使用另一个接口 - 同花顺
            if stock_data is None or stock_data.empty:
                try:
                    print(f"尝试使用同花顺数据源获取 {code} 数据")
                    # 对于沪市股票，需要添加前缀
                    ths_code = code
                    if code.startswith('6'):
                        ths_code = f"sh{code}"
                    elif code.startswith(('0', '3')):
                        ths_code = f"sz{code}"

                    stock_data = ak.stock_zh_a_hist_min_em(
                        symbol=ths_code,
                        period='daily',  # 使用日线数据
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq"
                    )

                    if stock_data is not None and not stock_data.empty:
                        successful_format = "stock_zh_a_hist_min_em"
                except Exception as e:
                    print(f"使用stock_zh_a_hist_min_em获取 {code} 失败: {e}")

            # 4. 尝试使用股票实时行情数据并记录
            if stock_data is None or stock_data.empty:
                try:
                    print(f"尝试获取 {code} 实时行情数据")
                    real_time_data = ak.stock_zh_a_spot_em()

                    # 筛选出目标股票
                    if '代码' in real_time_data.columns:
                        stock_real_time = real_time_data[real_time_data['代码'] == code]
                    elif '股票代码' in real_time_data.columns:
                        stock_real_time = real_time_data[real_time_data['股票代码'] == code]
                    else:
                        # 尝试其他可能的列名
                        for col in real_time_data.columns:
                            if 'code' in col.lower() or '代码' in col:
                                stock_real_time = real_time_data[real_time_data[col] == code]
                                break
                        else:
                            stock_real_time = pd.DataFrame()

                    if not stock_real_time.empty:
                        # 创建一个最小的历史数据框架
                        today = datetime.now().strftime('%Y-%m-%d')
                        stock_data = pd.DataFrame({
                            '日期': [today],
                            '股票代码': [code],
                            '股票名称': [name],
                            '开盘': [float(
                                stock_real_time['开盘'].iloc[0]) if '开盘' in stock_real_time.columns else np.nan],
                            '收盘': [float(
                                stock_real_time['最新价'].iloc[0]) if '最新价' in stock_real_time.columns else np.nan],
                            '最高': [float(
                                stock_real_time['最高'].iloc[0]) if '最高' in stock_real_time.columns else np.nan],
                            '最低': [float(
                                stock_real_time['最低'].iloc[0]) if '最低' in stock_real_time.columns else np.nan],
                            '成交量': [
                                float(stock_real_time['成交量'].iloc[0]) if '成交量' in stock_real_time.columns else 0],
                            '成交额': [
                                float(stock_real_time['成交额'].iloc[0]) if '成交额' in stock_real_time.columns else 0]
                        })

                        stock_data['日期'] = pd.to_datetime(stock_data['日期'])
                        stock_data.set_index('日期', inplace=True)
                        successful_format = "real_time_data"

                        print(f"注意: 仅获取到 {code} 的实时数据，不足以进行技术分析")
                except Exception as e:
                    print(f"获取 {code} 实时行情失败: {e}")

            # 所有方法都失败，放弃获取该股票数据
            if stock_data is None or stock_data.empty:
                print(f"无法获取股票 {code}-{name} 的数据，已尝试所有可能的方法")
                return None

            # 处理成功获取的数据
            print(f"成功使用{successful_format}获取 {code} 数据")
            print(f"获取的列名: {stock_data.columns.tolist()}")

            # 确保数据有日期索引
            if stock_data.index.name is None or not isinstance(stock_data.index, pd.DatetimeIndex):
                # 查找可能的日期列
                date_columns = ['日期', 'date', '时间', 'trade_date', 'datetime', 'Date', '日线日期']
                date_column = None

                for col in date_columns:
                    if col in stock_data.columns:
                        date_column = col
                        break

                # 如果找到日期列，设置为索引
                if date_column:
                    try:
                        stock_data[date_column] = pd.to_datetime(stock_data[date_column])
                        stock_data.set_index(date_column, inplace=True)
                    except Exception as e:
                        print(f"设置日期索引失败: {e}")
                else:
                    # 如果没有找到日期列，创建一个连续的日期索引
                    print("警告: 未找到日期列，创建虚拟日期索引")
                    stock_data['generated_date'] = pd.date_range(
                        start=pd.Timestamp(start_date),
                        periods=len(stock_data),
                        freq='D'
                    )
                    stock_data.set_index('generated_date', inplace=True)

            # 标准化列名
            column_mapping = {
                'open': '开盘', 'high': '最高', 'low': '最低', 'close': '收盘',
                'volume': '成交量', 'amount': '成交额', 'turnover': '换手率',
                'Open': '开盘', 'High': '最高', 'Low': '最低', 'Close': '收盘',
                'Volume': '成交量', 'Amount': '成交额', 'Turnover': '换手率'
            }

            # 重命名列
            for eng, chn in column_mapping.items():
                if eng in stock_data.columns and chn not in stock_data.columns:
                    stock_data[chn] = stock_data[eng]

            # 确保必要的列存在
            required_columns = ['开盘', '收盘', '最高', '最低', '成交量']
            for col in required_columns:
                if col not in stock_data.columns:
                    # 如果缺少价格列，尝试从其他价格列复制
                    if col in ['开盘', '最高', '最低'] and '收盘' in stock_data.columns:
                        stock_data[col] = stock_data['收盘']
                    elif col == '成交量' and col not in stock_data.columns:
                        stock_data[col] = 0
                    elif col == '收盘' and '开盘' in stock_data.columns:
                        stock_data[col] = stock_data['开盘']

            # 添加股票代码和名称列
            if '股票代码' not in stock_data.columns:
                stock_data['股票代码'] = code
            if '股票名称' not in stock_data.columns:
                stock_data['股票名称'] = name

            # 检查数据是否足够进行分析
            if len(stock_data) < 30:
                print(f"股票 {code} 数据不足30条 (实际: {len(stock_data)}条)，可能无法进行有效分析")
                if len(stock_data) < 5:  # 如果数据极少，返回None
                    return None

            # 如果启用缓存，保存数据
            if use_cache and not stock_data.empty:
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        # 准备用于JSON的数据
                        json_data = stock_data.reset_index()

                        # 处理日期列
                        for col in json_data.columns:
                            if pd.api.types.is_datetime64_any_dtype(json_data[col]):
                                json_data[col] = json_data[col].astype(str)

                        # 保存JSON
                        json.dump(json_data.to_dict('records'), f, ensure_ascii=False)
                except Exception as e:
                    print(f"保存缓存失败: {e}")

            return stock_data

        except Exception as e:
            print(f"获取 {code} 数据时发生未捕获的错误: {e}")
            traceback.print_exc()
            return None

    def process_stock(self, code, name, rsi_params=None):
        """处理单个股票的完整流程，支持自定义RSI参数"""
        # 获取股票数据
        data = self.get_stock_data(code, name)
        if data is None:
            return None

        # 计算技术指标，支持自定义RSI参数
        data_with_indicators = calculate_indicators(data, rsi_parameters=rsi_params)
        if data_with_indicators is None:
            return None

        # 保存处理后的数据
        self.stock_data[code] = data_with_indicators
        return data_with_indicators

    def parallel_process_stocks(self, max_workers=5, max_stocks=None, rsi_params=None):
        """并行处理多只股票，支持自定义RSI参数"""
        if self.all_stocks is None:
            self.get_all_stocks()

        # 可以限制处理的股票数量，用于测试
        stocks_to_process = self.all_stocks
        if max_stocks:
            stocks_to_process = self.all_stocks.head(max_stocks)

        print(f"开始处理 {len(stocks_to_process)} 只股票的数据...")

        # 使用线程池并行处理股票数据
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 创建任务列表
            future_to_stock = {
                executor.submit(self.process_stock, row['code'], row['name'], rsi_params):
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

    def screen_rsi_strategy(self,
                            rsi_period=14,
                            overbought_threshold=70,
                            oversold_threshold=30,
                            days_window=5,
                            mode='both',  # 'overbought', 'oversold', 'both'
                            require_trend_confirm=True,
                            min_strength=0):
        """
        增强的RSI策略筛选

        参数:
        rsi_period (int): 使用的RSI周期，如14
        overbought_threshold (float): 超买阈值
        oversold_threshold (float): 超卖阈值
        days_window (int): 检查的天数窗口
        mode (str): 筛选模式，'overbought'仅超买，'oversold'仅超卖，'both'两者都考虑
        require_trend_confirm (bool): 是否需要趋势确认（反转后的持续运动）
        min_strength (float): 最小背离强度，0-1之间

        返回:
        list: 筛选出的股票代码列表
        """
        rsi_key = f'rsi{rsi_period}'
        filtered = []
        results_details = {}

        for code, data in self.stock_data.items():
            # 检查数据是否足够
            if data is None or len(data) <= days_window + 2:
                continue

            try:
                # 获取最近数据
                recent_data = data.iloc[-days_window - 2:].copy()

                if len(recent_data) < 3:
                    continue

                # 检查是否存在RSI列
                if rsi_key not in recent_data.columns:
                    print(f"警告: {code} 缺少 {rsi_key} 列")
                    continue

                # 获取最新和前一天的RSI值
                latest_rsi = recent_data[rsi_key].iloc[-1]
                prev_rsi = recent_data[rsi_key].iloc[-2]

                # 获取收盘价列
                close_col = '收盘' if '收盘' in recent_data.columns else 'close'

                # 筛选逻辑
                selected = False
                reason = ""

                # 超买区域判断 (适用于趋势即将反转向下)
                if (mode in ['overbought', 'both'] and
                        overbought_threshold < latest_rsi < prev_rsi):  # RSI开始下降

                    if not require_trend_confirm:
                        selected = True
                        reason = f"RSI超买({latest_rsi:.2f})且开始下降"
                    else:
                        # 要求价格也开始下跌确认
                        if recent_data[close_col].iloc[-1] < recent_data[close_col].iloc[-2]:
                            selected = True
                            reason = f"RSI超买({latest_rsi:.2f})且价格开始下跌"

                # 超卖区域判断 (适用于趋势即将反转向上)
                elif (mode in ['oversold', 'both'] and
                      oversold_threshold > latest_rsi > prev_rsi):  # RSI开始上升

                    if not require_trend_confirm:
                        selected = True
                        reason = f"RSI超卖({latest_rsi:.2f})且开始上升"
                    else:
                        # 要求价格也开始上涨确认
                        if recent_data[close_col].iloc[-1] > recent_data[close_col].iloc[-2]:
                            selected = True
                            reason = f"RSI超卖({latest_rsi:.2f})且价格开始上涨"

                # 检查RSI背离（如果数据中包含背离信息）
                if 'rsi_divergence' in recent_data.columns:
                    last_divergence = recent_data['rsi_divergence'].iloc[-1]

                    # 如果是字典类型的背离数据
                    if isinstance(last_divergence, dict):
                        # 检查是否有足够强度的背离
                        if last_divergence.get('strength', 0) >= min_strength:
                            if mode in ['oversold', 'both'] and last_divergence.get('bullish', False):
                                selected = True
                                reason += f" 看涨背离(强度:{last_divergence['strength']:.2f})"
                            elif mode in ['overbought', 'both'] and last_divergence.get('bearish', False):
                                selected = True
                                reason += f" 看跌背离(强度:{last_divergence['strength']:.2f})"

                # 将选中的股票加入结果
                if selected:
                    filtered.append(code)

                    # 保存详细分析结果
                    results_details[code] = {
                        'rsi_value': latest_rsi,
                        'prev_rsi': prev_rsi,
                        'reason': reason,
                        'last_price': recent_data[close_col].iloc[-1],
                        'price_change': (recent_data[close_col].iloc[-1] / recent_data[close_col].iloc[-2] - 1) * 100
                    }

            except Exception as e:
                print(f"处理股票 {code} 的RSI数据时出错: {e}")
                continue

        # 保存详细结果以便后续分析
        self.rsi_screening_details = results_details

        return filtered

    def get_rsi_screening_details(self):
        """获取最近一次RSI筛选的详细结果"""
        if hasattr(self, 'rsi_screening_details'):
            return self.rsi_screening_details
        return {}

    def run_screening(self, max_stocks=None, strategy='rsi', days_ago=5, strategy_params=None):
        """
        运行完整的筛选流程，支持自定义参数

        参数:
        max_stocks (int): 最大处理股票数量，None表示处理所有
        strategy (str): 筛选策略
        days_ago (int): 查找信号的天数窗口
        strategy_params (dict): 策略特定参数
        """
        # 获取所有股票
        if self.all_stocks is None:
            self.get_all_stocks()
        print(f"strategy={strategy}")

        # 设置默认的策略参数
        default_params = {
            'rsi': {
                'rsi_period': 14,
                'overbought_threshold': 70,
                'oversold_threshold': 30,
                'mode': 'both',
                'require_trend_confirm': True,
                'min_strength': 0
            },
            'kdj': {},
            'macd': {},
            'combined': {}
        }

        # 合并默认参数和用户提供的参数
        params = default_params.get(strategy, {})
        if strategy_params:
            params.update(strategy_params)

        # 设置RSI计算参数（如果是RSI策略）
        rsi_calculate_params = None
        if strategy == 'rsi':
            rsi_calculate_params = {
                'periods': [params['rsi_period']],  # 使用策略指定的周期
                'smoothing_type': 'simple',
                'include_stochastic': True
            }

        # 处理股票数据
        self.parallel_process_stocks(max_stocks=max_stocks, rsi_params=rsi_calculate_params)

        # 根据选择的策略筛选股票
        if strategy == 'kdj':
            self.filtered_stocks = self.screen_kdj_golden_cross(days_ago)
        elif strategy == 'macd':
            self.filtered_stocks = self.screen_macd_golden_cross(days_ago)
        elif strategy == 'rsi':
            self.filtered_stocks = self.screen_rsi_strategy(
                rsi_period=params['rsi_period'],
                overbought_threshold=params['overbought_threshold'],
                oversold_threshold=params['oversold_threshold'],
                days_window=days_ago,
                mode=params['mode'],
                require_trend_confirm=params['require_trend_confirm'],
                min_strength=params['min_strength']
            )
        elif strategy == 'combined':
            self.filtered_stocks = self.screen_combined_strategy(days_ago)
        else:
            raise ValueError(f"不支持的策略: {strategy}")

        # 获取筛选出的股票的名称和详细信息
        filtered_with_info = []
        for code in self.filtered_stocks:
            try:
                name = self.all_stocks[self.all_stocks['code'] == code]['name'].iloc[0]
                details = {}

                # 添加详细原因（如果可用）
                if strategy == 'rsi' and hasattr(self, 'rsi_screening_details'):
                    details = self.rsi_screening_details.get(code, {})

                filtered_with_info.append({
                    'code': code,
                    'name': name,
                    **details
                })
            except:
                # 如果找不到名称，仅使用代码
                filtered_with_info.append({
                    'code': code,
                    'name': '未知',
                    'reason': '股票基本信息获取失败'
                })

        return filtered_with_info

    # 保留其他现有方法（screen_kdj_golden_cross, screen_macd_golden_cross等）
    def screen_kdj_golden_cross(self, days_window=5):
        """KDJ金叉策略筛选"""
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
        """MACD金叉策略筛选"""
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

    def screen_combined_strategy(self, days_window=5):
        """组合策略筛选"""
        # 获取各个单独策略的结果
        kdj_stocks = set(self.screen_kdj_golden_cross(days_window))
        macd_stocks = set(self.screen_macd_golden_cross(days_window))

        # 使用增强版RSI筛选，关注超卖反弹
        rsi_stocks = set(self.screen_rsi_strategy(
            days_window=days_window,
            mode='oversold',
            oversold_threshold=30
        ))

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