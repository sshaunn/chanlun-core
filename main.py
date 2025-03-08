from AStockScreener import AStockScreener


def check_akshare_functions():
    """检查akshare版本和可用的股票数据获取函数"""
    import akshare as ak
    import inspect

    # 获取版本
    version = ak.__version__ if hasattr(ak, '__version__') else "未知"
    print(f"当前akshare版本: {version}")

    # 获取可能与股票历史数据相关的函数
    stock_hist_funcs = []
    for name in dir(ak):
        if ('stock' in name.lower() and
                ('hist' in name.lower() or
                 'daily' in name.lower() or
                 'day' in name.lower())):
            stock_hist_funcs.append(name)

    print(f"找到 {len(stock_hist_funcs)} 个可能的股票历史数据函数:")
    for func in sorted(stock_hist_funcs):
        print(f"  - {func}")

    # 获取实时数据相关函数
    stock_spot_funcs = []
    for name in dir(ak):
        if 'stock' in name.lower() and ('spot' in name.lower() or 'real' in name.lower()):
            stock_spot_funcs.append(name)

    print(f"找到 {len(stock_spot_funcs)} 个可能的实时数据函数:")
    for func in sorted(stock_spot_funcs):
        print(f"  - {func}")

    return version, stock_hist_funcs, stock_spot_funcs


def test_stock_data_fetch():
    """
    测试股票数据获取的独立脚本
    """
    import akshare as ak
    import pandas as pd
    from datetime import datetime

    # 1. 打印akshare版本
    version = ak.__version__ if hasattr(ak, '__version__') else "未知"
    print(f"akshare版本: {version}")

    # 2. 测试几个常见的函数
    test_functions = [
        "stock_zh_a_hist",
        "stock_zh_a_daily"
    ]

    # 添加其他可能存在的函数
    for name in dir(ak):
        if ('stock' in name.lower() and
                ('hist' in name.lower() or 'daily' in name.lower())):
            if name not in test_functions:
                test_functions.append(name)

    # 检查这些函数是否存在
    for func_name in test_functions:
        exists = hasattr(ak, func_name)
        print(f"函数 {func_name} {'存在' if exists else '不存在'}")

    # 3. 测试不同的股票代码格式
    test_codes = ["000001", "sh000001", "sz000001", "600001", "sh600001"]
    test_func = "stock_zh_a_hist"  # 使用确认存在的函数

    if hasattr(ak, test_func):
        for code in test_codes:
            try:
                print(f"\n尝试用 {test_func} 获取 {code} 数据...")
                data = getattr(ak, test_func)(
                    symbol=code,
                    period="daily",
                    start_date="20240101",
                    end_date="20250110",
                    adjust="qfq"
                )
                if data is not None and not data.empty:
                    print(f"成功! 获取到 {len(data)} 条记录")
                    print(f"列名: {data.columns.tolist()}")
                    print(data.head(2))
                else:
                    print("失败! 返回空数据")
            except Exception as e:
                print(f"失败! 错误: {e}")

    # 4. 测试一个已知可用的股票
    working_codes = ["000409", "000504", "000509"]  # 您提到这些成功了
    for code in working_codes:
        try:
            print(f"\n尝试获取已知可用股票 {code} 数据...")
            data = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date="20240101",
                end_date="20250110",
                adjust="qfq"
            )
            if data is not None and not data.empty:
                print(f"成功! 获取到 {len(data)} 条记录")
                print(f"列名: {data.columns.tolist()}")
                print(data.head(2))
            else:
                print("失败! 返回空数据")
        except Exception as e:
            print(f"失败! 错误: {e}")


def analyze_stock_code_format():
    """分析什么样的股票代码格式能被成功处理"""
    import akshare as ak

    # 您提到的成功案例和失败案例
    success_codes = ["000409", "000504", "000509"]
    fail_codes = ["000004", "000007", "000008"]

    # 对比分析
    print("===== 成功股票代码分析 =====")
    for code in success_codes:
        print(f"代码: {code}")
        print(f"  - 长度: {len(code)}")
        print(f"  - 前缀: {code[:3]}")
        print(f"  - 是否为纯数字: {code.isdigit()}")

    print("\n===== 失败股票代码分析 =====")
    for code in fail_codes:
        print(f"代码: {code}")
        print(f"  - 长度: {len(code)}")
        print(f"  - 前缀: {code[:3]}")
        print(f"  - 是否为纯数字: {code.isdigit()}")

    # 尝试获取股票基本信息
    print("\n===== 获取股票列表 =====")
    try:
        all_stocks = ak.stock_info_a_code_name()
        print(f"共获取到 {len(all_stocks)} 只股票")
        print("前5只股票:")
        print(all_stocks.head())

        # 检查成功和失败的股票是否在列表中
        for code in success_codes + fail_codes:
            exists = code in all_stocks['code'].values
            print(f"股票 {code} {'存在' if exists else '不存在'}于股票列表中")
    except Exception as e:
        print(f"获取股票列表失败: {e}")

# 使用示例
if __name__ == "__main__":
    # check_akshare_functions()
    # test_stock_data_fetch()
    # analyze_stock_code_format()
    screener = AStockScreener()

    # 设置为None则处理所有股票
    max_stocks = 100  # 先用少量测试

    # 运行筛选，选择策略（'kdj', 'macd', 'rsi', 'boll', 'ema', 'combined'）
    # TODO boll ema is in processing
    filtered_stocks = screener.run_screening(max_stocks=max_stocks, strategy='rsi', days_ago=5)

    print("\n符合筛选条件的股票:")
    for code, name in filtered_stocks:
        print(f"{code} - {name}")