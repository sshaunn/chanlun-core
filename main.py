from AStockScreener import AStockScreener


def check_akshare_functions():
    """检查akshare版本和可用的股票数据获取函数"""
    # import akshare as ak
    import akshare as ak
    import inspect

    # 获取版本
    version = ak.__version__ if hasattr(ak, '__version__') else "未知"
    print(f"当前akshare版本: {version}")

# 使用示例
if __name__ == "__main__":
    # TODO remove when no need anymore
    check_akshare_functions()

    screener = AStockScreener()
    # 设置为None则处理所有股票
    max_stocks = 100  # 先用少量测试

    # 运行筛选，选择策略（'kdj', 'macd', 'rsi', 'boll', 'ema', 'combined'）
    # TODO boll ema is in processing
    filtered_stocks = screener.run_screening(max_stocks=max_stocks, strategy='rsi', days_ago=5)

    print("\n符合筛选条件的股票:")
    for stock_info in filtered_stocks:
        print(f"{stock_info}")