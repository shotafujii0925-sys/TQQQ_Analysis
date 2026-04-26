import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats

def analyze_statistical_significance(period="3mo"):
    # データ取得
    tickers = ["TQQQ", "SPY"]
    data = yf.download(tickers, period=period)['Close'].ffill()
    
    # 日次リターンの計算
    returns = data.pct_change().dropna()
    
    # 1. 基本統計量の計算
    stats_dict = {}
    for col in tickers:
        daily_mean = returns[col].mean()
        daily_std = returns[col].std()
        # 年率換算 (252営業日)
        ann_return = daily_mean * 252
        ann_vol = daily_std * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol != 0 else 0
        
        # 最大下落率 (Max Drawdown)
        cumulative = (1 + returns[col]).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()
        
        stats_dict[col] = {
            "年率リターン": ann_return,
            "年率ボラティリティ": ann_vol,
            "シャープレシオ": sharpe,
            "最大下落率": max_dd
        }
    
    # 2. 統計的検定 (対応のあるt検定)
    t_stat, p_value = stats.ttest_rel(returns['TQQQ'], returns['SPY'])
    
    # 結果の表示
    res_df = pd.DataFrame(stats_dict).T
    print("--- 投資指標比較 ---")
    print(res_df)
    print(f"\n--- 統計検定 (t検定) ---")
    print(f"t値: {t_stat:.4f}")
    print(f"p値: {p_value:.4f}")
    
    if p_value > 0.05:
        print("\n【結論】p値が0.05を超えているため、統計的有意差は認められません。")
    else:
        print("\n【結論】統計的有意差が認められました。")

if __name__ == "__main__":
    analyze_statistical_significance(period="3mo")