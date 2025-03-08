# util.py
import json
import os
import traceback
from datetime import datetime, timedelta

import akshare as ak
import numpy as np
import pandas as pd


def get_stock_data(code, name=None, period="daily", start_date="20220101", end_date="20250101",
                   use_cache=True, cache_dir="./stock_cache"):
    """
    获取单个股票的历史数据 - 针对akshare优化的独立函数版本

    参数:
    code (str): 股票代码，如'000001'
    name (str, optional): 股票名称，用于日志和缓存。如果为None，将使用代码作为名称
    period (str): 数据周期，默认为'daily'
    start_date (str): 开始日期，格式为'YYYYMMDD'
    end_date (str): 结束日期，格式为'YYYYMMDD'
    use_cache (bool): 是否使用缓存
    cache_dir (str): 缓存目录路径

    返回:
    pandas.DataFrame: 包含股票历史数据的DataFrame，或None（如果获取失败）
    """
    # 确保缓存目录存在
    if use_cache and not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir)
        except Exception as e:
            print(f"创建缓存目录失败: {e}")
            use_cache = False  # 如果无法创建目录，禁用缓存

    # 如果没有提供股票名称，使用代码作为名称
    if name is None:
        name = code

    # 缓存文件路径
    cache_file = os.path.join(cache_dir, f"{code}_data.json")

    # 尝试从缓存加载数据
    if use_cache and os.path.exists(cache_file):
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - file_time < timedelta(days=1):
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
                                print(f"从缓存加载股票 {code} 数据成功")
                                return stock_data
                        print(f"缓存数据中未找到有效的日期列: {stock_data.columns.tolist()}")
            except Exception as e:
                print(f"读取缓存出错: {e}")

    # 尝试所有可能的数据获取方法
    try:
        # 1. 使用stock_zh_a_hist_min_em聚合为日线数据 (最可靠的方法)
        try:
            print(f"尝试使用stock_zh_a_hist_min_em获取股票 {code} 数据...")

            # 根据股票代码确定正确的前缀
            em_code = code
            if code.startswith('6'):
                em_code = f"sh{code}"
            elif code.startswith(('0', '3')):
                em_code = f"sz{code}"

            # 使用分钟线数据接口
            min_data = ak.stock_zh_a_hist_min_em(
                symbol=em_code,
                period='1',  # 1分钟线
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )

            if min_data is not None and not min_data.empty:
                print(f"成功获取分钟线数据，共 {len(min_data)} 条记录")
                print(f"列名: {min_data.columns.tolist()}")

                # 确定日期列
                time_col = None
                for col in ['时间', '日期', 'datetime', 'date']:
                    if col in min_data.columns:
                        time_col = col
                        break

                if time_col is None:
                    raise ValueError("未找到分钟线数据中的时间列")

                # 转换时间列为datetime类型
                min_data[time_col] = pd.to_datetime(min_data[time_col])

                # 提取日期部分
                min_data['trade_date'] = min_data[time_col].dt.date

                # 按日期聚合，计算OHLC
                agg_dict = {}

                # 确定价格列和交易量列
                price_vol_cols = {
                    '开盘': 'first',
                    '收盘': 'last',
                    '最高': 'max',
                    '最低': 'min',
                    '成交量': 'sum',
                    '成交额': 'sum',
                    'open': 'first',
                    'close': 'last',
                    'high': 'max',
                    'low': 'min',
                    'volume': 'sum',
                    'amount': 'sum'
                }

                # 添加存在的列到聚合字典
                for col, agg_func in price_vol_cols.items():
                    if col in min_data.columns:
                        agg_dict[col] = agg_func

                # 如果没有找到任何可聚合的列
                if not agg_dict:
                    raise ValueError(f"未找到可聚合的价格或交易量列: {min_data.columns}")

                # 执行聚合
                stock_data = min_data.groupby('trade_date').agg(agg_dict).reset_index()

                # 将日期转换为datetime类型
                stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
                stock_data.set_index('trade_date', inplace=True)

                print(f"成功聚合为日线数据，共 {len(stock_data)} 个交易日")

                # 标准化列名
                column_mapping = {
                    'open': '开盘', 'high': '最高', 'low': '最低', 'close': '收盘',
                    'volume': '成交量', 'amount': '成交额', 'turnover': '换手率'
                }

                # 应用标准列名映射
                for eng, chn in column_mapping.items():
                    if eng in stock_data.columns and chn not in stock_data.columns:
                        stock_data[chn] = stock_data[eng]

                # 确保至少有价格列
                for col in ['开盘', '收盘', '最高', '最低']:
                    if col not in stock_data.columns:
                        for ref_col in ['开盘', '收盘', '最高', '最低']:
                            if ref_col in stock_data.columns:
                                stock_data[col] = stock_data[ref_col]
                                break
                        else:
                            # 如果找不到任何价格列作为参考
                            stock_data[col] = np.nan

                # 添加股票代码和名称
                stock_data['股票代码'] = code
                stock_data['股票名称'] = name

                # 如果使用缓存，保存数据
                if use_cache:
                    try:
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            json_data = stock_data.reset_index()
                            # 转换日期列为字符串
                            for col in json_data.columns:
                                if pd.api.types.is_datetime64_any_dtype(json_data[col]):
                                    json_data[col] = json_data[col].astype(str)
                            json.dump(json_data.to_dict('records'), f, ensure_ascii=False)
                    except Exception as e:
                        print(f"保存缓存失败: {e}")

                return stock_data

        except Exception as e:
            print(f"获取并聚合分钟线数据失败: {e}")
            traceback.print_exc()

        # 2. 尝试使用stock_zh_a_hist_tx函数
        try:
            print(f"尝试使用stock_zh_a_hist_tx获取股票 {code} 数据...")

            # 腾讯接口可能使用不同的股票代码格式
            tx_code = code

            stock_data = ak.stock_zh_a_hist_tx(symbol=tx_code)

            if stock_data is not None and not stock_data.empty:
                print(f"成功使用stock_zh_a_hist_tx获取数据，共 {len(stock_data)} 条记录")
                print(f"列名: {stock_data.columns.tolist()}")

                # 确定日期列
                date_column = None
                for col in ['date', '日期', '时间', 'trade_date', 'datetime']:
                    if col in stock_data.columns:
                        date_column = col
                        break

                # 如果找到日期列，设置为索引
                if date_column:
                    stock_data[date_column] = pd.to_datetime(stock_data[date_column])
                    stock_data.set_index(date_column, inplace=True)
                else:
                    # 使用行索引作为日期
                    stock_data['generated_date'] = pd.date_range(
                        start=pd.Timestamp(start_date),
                        periods=len(stock_data),
                        freq='D'
                    )
                    stock_data.set_index('generated_date', inplace=True)

                # 标准化列名和添加股票信息
                column_mapping = {
                    'open': '开盘', 'high': '最高', 'low': '最低', 'close': '收盘',
                    'volume': '成交量', 'amount': '成交额', 'turnover': '换手率'
                }

                # 应用标准列名映射
                for eng, chn in column_mapping.items():
                    if eng in stock_data.columns and chn not in stock_data.columns:
                        stock_data[chn] = stock_data[eng]

                # 确保有基本价格列
                for col in ['开盘', '收盘', '最高', '最低']:
                    if col not in stock_data.columns:
                        for ref_col in ['开盘', '收盘', '最高', '最低']:
                            if ref_col in stock_data.columns:
                                stock_data[col] = stock_data[ref_col]
                                break
                        else:
                            stock_data[col] = np.nan

                # 添加股票代码和名称
                stock_data['股票代码'] = code
                stock_data['股票名称'] = name

                # 如果使用缓存，保存数据
                if use_cache:
                    try:
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            json_data = stock_data.reset_index()
                            # 转换日期列为字符串
                            for col in json_data.columns:
                                if pd.api.types.is_datetime64_any_dtype(json_data[col]):
                                    json_data[col] = json_data[col].astype(str)
                            json.dump(json_data.to_dict('records'), f, ensure_ascii=False)
                    except Exception as e:
                        print(f"保存缓存失败: {e}")

                return stock_data

        except Exception as e:
            print(f"使用stock_zh_a_hist_tx获取数据失败: {e}")

        # 3. 尝试使用stock_zh_a_daily
        try:
            print(f"尝试使用stock_zh_a_daily获取股票 {code} 数据...")

            # 现在我们知道stock_zh_a_daily可能不接受日期参数，可能需要手动截取
            stock_data_full = ak.stock_zh_a_daily(symbol=code)

            # 如果成功获取数据
            if stock_data_full is not None and not stock_data_full.empty:
                print(f"成功获取到全部数据，共 {len(stock_data_full)} 条记录")
                print(f"列名: {stock_data_full.columns.tolist()}")

                # 检查是否有日期列
                date_column = None
                for col in ['date', '日期', '时间', 'trade_date', 'datetime']:
                    if col in stock_data_full.columns:
                        date_column = col
                        break

                # 如果找到日期列，筛选日期范围
                if date_column:
                    try:
                        # 转换日期列和筛选条件
                        stock_data_full[date_column] = pd.to_datetime(stock_data_full[date_column])
                        start = pd.Timestamp(start_date)
                        end = pd.Timestamp(end_date)

                        # 筛选日期范围内的数据
                        stock_data = stock_data_full[
                            (stock_data_full[date_column] >= start) &
                            (stock_data_full[date_column] <= end)
                            ]

                        # 设置日期为索引
                        stock_data.set_index(date_column, inplace=True)
                    except Exception as date_e:
                        print(f"日期筛选失败: {date_e}")
                        stock_data = stock_data_full
                else:
                    # 没有日期列，使用全部数据
                    stock_data = stock_data_full
                    # 创建日期索引
                    stock_data['generated_date'] = pd.date_range(
                        start=pd.Timestamp('20200101'),
                        periods=len(stock_data),
                        freq='D'
                    )
                    stock_data.set_index('generated_date', inplace=True)

                # 标准化列名
                column_mapping = {
                    'open': '开盘', 'high': '最高', 'low': '最低', 'close': '收盘',
                    'volume': '成交量', 'amount': '成交额', 'turnover': '换手率'
                }

                # 应用标准列名映射
                for eng, chn in column_mapping.items():
                    if eng in stock_data.columns and chn not in stock_data.columns:
                        stock_data[chn] = stock_data[eng]

                # 确保有基本价格列
                for col in ['开盘', '收盘', '最高', '最低']:
                    if col not in stock_data.columns:
                        for ref_col in ['开盘', '收盘', '最高', '最低']:
                            if ref_col in stock_data.columns:
                                stock_data[col] = stock_data[ref_col]
                                break
                        else:
                            stock_data[col] = np.nan

                # 添加股票代码和名称
                stock_data['股票代码'] = code
                stock_data['股票名称'] = name

                # 如果使用缓存，保存数据
                if use_cache:
                    try:
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            json_data = stock_data.reset_index()
                            # 转换日期列为字符串
                            for col in json_data.columns:
                                if pd.api.types.is_datetime64_any_dtype(json_data[col]):
                                    json_data[col] = json_data[col].astype(str)
                            json.dump(json_data.to_dict('records'), f, ensure_ascii=False)
                    except Exception as e:
                        print(f"保存缓存失败: {e}")

                return stock_data

        except Exception as e:
            print(f"使用stock_zh_a_daily获取数据失败: {e}")

            # 检查是否是'date'键错误，这意味着函数实现可能改变了
            if "'date'" in str(e):
                try:
                    # 尝试修改函数源码查看问题
                    import inspect
                    func_code = inspect.getsource(ak.stock_zh_a_daily)
                    print(f"stock_zh_a_daily函数源码:\n{func_code}")
                except:
                    pass

        # 4. 尝试获取实时数据，仅作为最后的备选方案
        try:
            print(f"尝试获取 {code} 的实时行情数据...")

            # 尝试几个可能的实时数据函数
            real_time_data = None

            # 首先尝试 stock_zh_a_spot_em
            try:
                real_time_data = ak.stock_zh_a_spot_em()
            except Exception as e1:
                print(f"使用stock_zh_a_spot_em获取实时数据失败: {e1}")

                # 然后尝试 stock_zh_a_spot
                try:
                    real_time_data = ak.stock_zh_a_spot()
                except Exception as e2:
                    print(f"使用stock_zh_a_spot获取实时数据失败: {e2}")

            if real_time_data is not None and not real_time_data.empty:
                print(f"成功获取实时数据，共 {len(real_time_data)} 条记录")
                print(f"列名: {real_time_data.columns.tolist()}")

                # 查找代码列
                code_column = None
                for col in ['代码', '股票代码', 'code', 'symbol']:
                    if col in real_time_data.columns:
                        code_column = col
                        break

                if code_column is None:
                    raise ValueError("未找到实时数据中的股票代码列")

                # 筛选目标股票
                stock_real_time = real_time_data[real_time_data[code_column] == code]

                if stock_real_time.empty:
                    # 尝试其他可能的代码格式
                    if code.startswith('6'):
                        alt_codes = [f"sh{code}", f"{code}.SH"]
                    else:
                        alt_codes = [f"sz{code}", f"{code}.SZ"]

                    for alt_code in alt_codes:
                        stock_real_time = real_time_data[real_time_data[code_column] == alt_code]
                        if not stock_real_time.empty:
                            break

                if not stock_real_time.empty:
                    print(f"在实时数据中找到股票 {code}")

                    # 找出价格和交易量列
                    col_map = {
                        '开盘': ['开盘', '开盘价', 'open'],
                        '收盘': ['最新价', '现价', '收盘', 'close', 'price'],
                        '最高': ['最高', '最高价', 'high'],
                        '最低': ['最低', '最低价', 'low'],
                        '成交量': ['成交量', 'volume', 'vol'],
                        '成交额': ['成交额', 'amount']
                    }

                    data_dict = {'股票代码': code, '股票名称': name}

                    # 从实时数据中提取各项值
                    for target_col, possible_cols in col_map.items():
                        for possible_col in possible_cols:
                            if possible_col in stock_real_time.columns:
                                try:
                                    data_dict[target_col] = float(stock_real_time[possible_col].iloc[0])
                                    break
                                except:
                                    pass
                        if target_col not in data_dict:
                            # 如果找不到该值，尝试使用其他价格列
                            for price_col in ['收盘', '开盘', '最高', '最低']:
                                if price_col in data_dict:
                                    data_dict[target_col] = data_dict[price_col]
                                    break
                            else:
                                # 仍找不到，使用默认值
                                if target_col in ['成交量', '成交额']:
                                    data_dict[target_col] = 0
                                else:
                                    data_dict[target_col] = np.nan

                    # 创建单行数据
                    today = datetime.now().strftime('%Y-%m-%d')
                    data_dict['日期'] = today

                    # 转换为DataFrame
                    stock_data = pd.DataFrame([data_dict])
                    stock_data['日期'] = pd.to_datetime(stock_data['日期'])
                    stock_data.set_index('日期', inplace=True)

                    print(f"成功创建单日数据: \n{stock_data}")

                    # 如果使用缓存，保存数据
                    if use_cache:
                        try:
                            with open(cache_file, 'w', encoding='utf-8') as f:
                                json_data = stock_data.reset_index()
                                # 转换日期列为字符串
                                for col in json_data.columns:
                                    if pd.api.types.is_datetime64_any_dtype(json_data[col]):
                                        json_data[col] = json_data[col].astype(str)
                                json.dump(json_data.to_dict('records'), f, ensure_ascii=False)
                        except Exception as e:
                            print(f"保存缓存失败: {e}")

                    # 警告：单日数据不足以进行技术分析
                    print(f"警告：仅获取到单日数据，不足以进行复杂的技术分析")

                    return stock_data
                else:
                    print(f"在实时数据中未找到股票 {code}")

        except Exception as e:
            print(f"获取实时数据失败: {e}")

        # 5. 最后尝试创建一个最小的股票数据集，从股票基本信息推导
        try:
            print(f"尝试获取股票 {code} 的基本信息...")

            try:
                # 尝试获取股票基本信息
                stock_info = None

                # 尝试不同的函数获取股票信息
                try:
                    if hasattr(ak, 'stock_individual_info_em'):
                        stock_info = ak.stock_individual_info_em(symbol=code)
                except:
                    pass

                if stock_info is None:
                    try:
                        if hasattr(ak, 'stock_info_a_code_name'):
                            all_stocks = ak.stock_info_a_code_name()
                            stock_info = all_stocks[all_stocks['code'] == code]
                    except:
                        pass

                if stock_info is not None and not (hasattr(stock_info, 'empty') and stock_info.empty):
                    print(f"成功获取到股票 {code} 的基本信息")

                    # 创建一个最小的数据集
                    min_data = pd.DataFrame({
                        '日期': pd.date_range(start=pd.Timestamp(start_date), periods=30, freq='D'),
                        '股票代码': [code] * 30,
                        '股票名称': [name] * 30,
                        '开盘': np.nan,
                        '收盘': np.nan,
                        '最高': np.nan,
                        '最低': np.nan,
                        '成交量': 0
                    })

                    # 设置日期为索引
                    min_data.set_index('日期', inplace=True)

                    print(f"创建了最小占位数据集，共 {len(min_data)} 条记录")

                    # 如果使用缓存，保存数据
                    if use_cache:
                        try:
                            with open(cache_file, 'w', encoding='utf-8') as f:
                                json_data = min_data.reset_index()
                                # 转换日期列为字符串
                                for col in json_data.columns:
                                    if pd.api.types.is_datetime64_any_dtype(json_data[col]):
                                        json_data[col] = json_data[col].astype(str)
                                json.dump(json_data.to_dict('records'), f, ensure_ascii=False)
                        except Exception as e:
                            print(f"保存缓存失败: {e}")

                    print(f"警告：创建了填充数据，实际上没有获取到真实的历史价格数据")

                    # 返回最小数据集，但标记为不可用于技术分析
                    min_data['数据可用'] = False

                    return min_data

            except Exception as e_info:
                print(f"获取股票基本信息失败: {e_info}")

        except Exception as e:
            print(f"创建最小数据集失败: {e}")

        # 所有方法都失败，返回None
        print(f"所有方法都失败，无法获取股票 {code} 的数据")
        return None

    except Exception as e:
        print(f"获取股票 {code} 数据时发生未捕获错误: {e}")
        traceback.print_exc()
        return None


def check_akshare_version():
    """检查当前akshare版本"""
    try:
        import akshare as ak
        version = getattr(ak, '__version__', 'unknown')
        print(f"当前akshare版本: {version}")
        return version
    except Exception as e:
        print(f"检查akshare版本失败: {e}")
        return "unknown"