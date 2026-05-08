import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint


#get data:
def get_data(tickers,start_date,end_date):
    data=yf.download(tickers,start=start_date,end=end_date)['Close']
    return data.dropna()

#cointegered pairs
def get_pairs(data):
    n=data.shape[1]
    keys=data.columns
    pairs=[]

    for i in range(n):
        for j in range(i+1,n):
            s1,s2=data[keys[i]],data[keys[j]]
            result= coint(s1,s2)
            pvalue=result[1]

            if pvalue<0.05 :
                pairs.append((keys[i],keys[j],pvalue))
    return pairs

#finding spread:
def spread(s1,s2,window=20):
    x=s1.values.reshape(-1,1)
    y=s2.values

    model=LinearRegression()
    model.fit(x,y)

    beta=model.coef_[0]
    intercept=model.intercept_

    spread=s2-(beta*s1+intercept)  

    mean=spread.rolling(window=window).mean()   
    std=spread.rolling(window=window).std()
    z_score=(spread-mean)/std
    return z_score,beta


def main():
    user_input=input("enter stickers seperated by commas\n")
    tickers= [t.strip().upper() for t in user_input.split(',')]

    start_date = "2024-01-01"
    end_date = "2026-01-01"

    df=get_data(tickers,start_date,end_date)

    if df.empty or len(df.columns)<2:
        print("empty\n")
        return
    
    pairs=get_pairs(df)
    if not pairs:
        print("no pairs found")
    else:
        print("pairs found")
        for p in pairs:
            print(f"Pair: {p[0]} & {p[1]} | P-Value: {p[2]:.4f}")

            return
        

    best_p = min(pairs, key=lambda x: x[2])
    s1, s2 = best_p[0], best_p[1]
    z_score, hedge_ratio = spread(df[s1], df[s2])


    # plt.figure(figsize=(12, 6))
    # z_score.plot(label=f'Z-Score ({s1} vs {s2})', color='blue')
    # plt.axhline(2.0, color='red', linestyle='--', label='Short Threshold')
    # plt.axhline(-2.0, color='green', linestyle='--', label='Long Threshold')
    # plt.axhline(0, color='black', alpha=0.5)
    # plt.title(f"StatArb Signals: {s1} & {s2} (Hedge Ratio: {hedge_ratio:.2f})")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show()

    # import matplotlib.pyplot as plt

   # 1. Prepare the Canvas
    plt.figure(figsize=(14, 7), facecolor='#f4f4f4')

# 2. Plot the Z-Score line
    z_score.plot(label=f'Z-Score ({s1} vs {s2})', color='#1f77b4', linewidth=1.5, zorder=1)

# 3. Add Trade Signal Markers
# Finding where Z-score crosses the thresholds
    buys = z_score[z_score <= -2.0]
    sells = z_score[z_score >= 2.0]

    plt.scatter(buys.index, buys.values, color='green', marker='^', label='Buy Signal', alpha=1, zorder=2)
    plt.scatter(sells.index, sells.values, color='red', marker='v', label='Sell Signal', alpha=1, zorder=2)

# 4. Add Horizontal Threshold Lines
    plt.axhline(2.0, color='red', linestyle='--', linewidth=1, label='Short Threshold (+2.0)')
    plt.axhline(-2.0, color='green', linestyle='--', linewidth=1, label='Long Threshold (-2.0)')
    plt.axhline(0, color='black', linewidth=0.8, alpha=0.5)

# 5. Shaded "Danger Zones"
    plt.axhspan(2.0, z_score.max() if z_score.max() > 2 else 3, color='red', alpha=0.05)
    plt.axhspan(-2.0, z_score.min() if z_score.min() < -2 else -3, color='green', alpha=0.05)

# 6. Formatting & Labels
    plt.title(f"Statistical Arbitrage Signals: {s1} vs {s2}\nSpread Hedge Ratio: {hedge_ratio:.4f}", 
          fontsize=14, fontweight='bold', pad=20)
    plt.ylabel("Z-Score (Standard Deviations)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)

# 7. Final Show Command
    plt.tight_layout()
    print(f">>> Displaying Graph for {s1} and {s2}...")
    plt.show()

if __name__ == "__main__":
    main()    
       
     
            



