# -- coding: utf-8 --
# @Time : 2022/2/7 20:31
# @Author : zhuo.wang
# @File : rsi.py
import warnings,os

import pandas as pd

warnings.filterwarnings('ignore')
import talib as ta
from multiprocessing import Pool


from tqdm import tqdm


width = 30/50 * 60
min_slope = 0.65
interval = 7
window = 40
sell_interval = 2




def get_slope(data):
    return ((data - data.shift(-interval)) / width).shift(interval)

def rsi_min(data,stock):
    slope = get_slope(data)
    min_rsi = data.min()
    max_slope = slope.max()
    rsi_value = data.iloc[-1]
#     if data.index[-1] == 604:
#         print(slope.idxmax(),max_slope,slope.iloc[-1],
#           (slope.iloc[-1] / max_slope),stock.loc[data.index[-1],'date'],
#               min_rsi,data.iloc[-1],data.iloc[-1],min_rsi
#          )
    before = min([window ,65])
    if data.index[-1] > before and rsi_value <= stock.loc[data.index[-1] - before:data.index[-1],'rsi'].min():
        return True
    if (slope.iloc[-1] / max_slope) >= 0.8 and max_slope >= min_slope and  rsi_value <= 40:
        if (rsi_value <= min_rsi *1.10) or rsi_value <= 10:
            return True
    return False

def run(stock,symbol):
    rsi = stock.rsi.rolling(window=window).apply(rsi_min,args=(stock,))
    stock['rsi_value'] = list(rsi)
    rsi_index = stock.loc[stock.rsi_value == 1].index
    resp = []
    latest_list = []
    for rsi_ix in rsi_index:
        try:
            buy_price = stock.loc[rsi_ix + 1, 'open']
            sell_price = stock.loc[rsi_ix + 1 + sell_interval, 'open']
        except KeyError:
            latest_list.append([symbol, stock.loc[rsi_ix, 'date']])
            continue
        cache = stock.loc[rsi_ix:rsi_ix]
        cache['profits'] = 1 - buy_price / sell_price
        resp.append(cache)
    return resp,latest_list


if __name__ == '__main__':
    from common.utils import load_df, BASE_DIR
    df = load_df()
    resp = []
    latest_list = []
    df2 = df.groupby('symbol').apply(lambda x: x.assign(rsi=ta.RSI(x.close, 6)))
    df2 = df2.dropna(subset=['rsi'])[['date', 'close', 'rsi', 'name', 'symbol', 'open']].reset_index(drop=True)
    t = tqdm(range(len(df2.loc[~df2.name.str.contains('ST')].symbol.unique())), ncols=70)
    pool = Pool(5)
    jobs = []
    for symbol, stock in df2.loc[~df2.name.str.contains('ST')].groupby('symbol'):
        jobs.append(pool.apply_async(run, args=(stock, symbol)))
    for job in jobs:
        data, latest_data = job.get()
        resp.extend(data)
        latest_list.extend(latest_data)
        t.update(1)

    pd.concat(resp).to_pickle(os.path.join(BASE_DIR, 'results', 'rsi.pkl'))
    pd.DataFrame(latest_list, columns=['symbol', 'date']).to_pickle(os.path.join(BASE_DIR, 'results', 'rsi_latest.pkl'))
