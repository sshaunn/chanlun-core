from AStockScreener import AStockScreener, calculate_indicators, get_stock_data

# 使用示例
if __name__ == "__main__":
    screener = AStockScreener()

    # 设置为None则处理所有股票
    max_stocks = 100  # 先用少量测试

    # 运行筛选，选择策略（'kdj', 'macd', 'rsi', 'boll', 'ema', 'combined'）
    # TODO boll ema is in processing
    filtered_stocks = screener.run_screening(max_stocks=max_stocks, strategy='rsi', days_ago=5)

    print("\n符合筛选条件的股票:")
    for code, name in filtered_stocks:
        print(f"{code} - {name}")