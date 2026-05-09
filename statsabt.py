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

    

    print("\n" + "=" * 50)
    print(f"Best Pair  : {s1} vs {s2}")
    print(f"Beta       : {hedge_ratio:.4f}")
    print("-" * 50)
    print("Z-Score (last 10 values):")
    print(z_score.dropna().tail(10).to_string())
    print("=" * 50 + "\n")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(z_score.index, z_score.values, color='blue', label=f'Z-Score ({s1} vs {s2})')
    ax1.axhline( 2.0, color='red',   linestyle='--', linewidth=1.2, label='Short Threshold (+2)')
    ax1.axhline(-2.0, color='green', linestyle='--', linewidth=1.2, label='Long Threshold  (-2)')
    ax1.axhline( 0,   color='black', linestyle='-',  alpha=0.4,     label='Mean (0)')
    ax1.fill_between(z_score.index, 2, z_score.values,
                     where=(z_score.values > 2),  alpha=0.1, color='red',   label='Short Zone')
    ax1.fill_between(z_score.index, -2, z_score.values,
                     where=(z_score.values < -2), alpha=0.1, color='green', label='Long Zone')
    ax1.set_title(f"Z-Score — {s1} vs {s2}", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Z-Score")
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)


    ax2.axhline(hedge_ratio, color='orange', linewidth=2,
                label=f'Hedge Ratio (β) = {hedge_ratio:.4f}')
    ax2.set_title(f"Hedge Ratio (Beta) — {s1} vs {s2}", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Beta")
    ax2.set_xlabel("Date")
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"StatArb Analysis: {s1} & {s2}", fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("statarb_signals.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Chart saved → statarb_signals.png")    
    
if __name__ == "__main__":
    main()    
       
     
            



