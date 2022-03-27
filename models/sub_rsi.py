# -- coding: utf-8 --
# @Time : 2022/3/5 19:29
# @Author : zhuo.wang
# @File : sub_ris.py
import datetime
import os
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

from tqdm import tqdm
import talib as ta

interval = 5
min_slope = 0.58
min_rsi_value = 42
width = 30 / 50 * 60
min_rsi_window = 30

class Trading:

    def __init__(self,money=90000):
        self.money = money
        self.local_money = money
        self.end_date = None
        self.use_day = 0

    @classmethod
    def add_rsi(self,data):
        for freq in [6, 12, 24]:
            data[f"rsi_{freq}"] = ta.RSI(data.close, freq)
        return data

    def get_slope(self,data):
        return ((data - data.shift(-interval)) / width).shift(interval).iloc[-1]

    def get_next_p_change(self,item,date, n=1):
        n = n + 1
        s = item.loc[date:].iloc[n]
        self.end_date = s.date
        return s.p_change / 100

    def get_next_date(self,item,date, n=1):
        return item.loc[date:].iloc[n].date

    def sub_money(self,buy_money, p_change):
        buy_money *= (1 + p_change)
        buy_money += self.local_money * (1 / 3)
        self.money -= self.local_money * (1 / 3)
        return buy_money

    def get_min_difference(self,data, tolerance):
        max_reduce = data.max()
        current_value = data.iloc[-1]
        if current_value >= max_reduce * tolerance:
            return True
        return False

    def double_sell(self,item,buy_money, date, n=3):
        profit = buy_money * 0.5
        buy_money -= profit
        profit += buy_money * (1 + self.get_next_p_change(item,date, n))
        self.money += profit

    def get_min_rsi(self,data):
        min_rsi = data.min()
        current_value = data.iloc[-1]
        return (current_value - min_rsi) / current_value <= 0.10 and current_value < 38

    def get_profit(self,date,item,symbol):
        self.use_day = 0
        self.end_date = date
        buy_money = self.money * (1 / 3)
        self.money -= buy_money
        p_change = self.get_next_p_change(item, date, 1)

        if p_change > 0:
            buy_money *= (1 + p_change)
            # self.double_sell(item, buy_money, date, n=2)

            buy_money *= (1 + self.get_next_p_change(item,date,n=2))
            self.money += buy_money

            self.use_day += 2
        else:
            buy_money = self.sub_money(buy_money, p_change)
            p_change = self.get_next_p_change(item, date, 2)
            self.use_day += 1
            if p_change > 0:
                buy_money *= (1 + p_change)
                self.double_sell(item, buy_money, date, n=3)
                self.use_day += 2
            else:
                buy_money = self.sub_money(buy_money, p_change)
                buy_money *= (1 + self.get_next_p_change(item, date, n=3))
                self.double_sell(item, buy_money, date, n=4)
                self.use_day += 2
        profit = self.money - self.local_money
        ratio = profit / buy_money
        self.money = self.local_money
        return [symbol,date, profit,ratio,buy_money,self.use_day, self.end_date]

    @classmethod
    def load_stock(cls):
        file_path = os.path.join(BASE_DIR, 'results', f'{datetime.datetime.now().date()}_sub_rsi_stock_data.pkl')
        if os.path.exists(file_path):
            return pd.read_pickle(file_path)
        df = load_df()
        df = df.groupby('symbol').apply(lambda x: Trading.add_rsi(x)).reset_index(drop=True)
        stock = df[['name', 'date', 'ts_code', 'symbol', 'open', 'close', 'high', 'low', 'rsi_6', 'rsi_12', 'rsi_24',
                    'p_change']]
        stock = stock.groupby('symbol').apply(lambda x: x.assign(reduce=x['rsi_24'] - x['rsi_6'])).reset_index(
            drop=True)
        stock.to_pickle(file_path)
        return stock

    @classmethod
    def save(cls,result,latest_result):
        total = pd.DataFrame(result, columns=['symbol', 'date', 'profit','ratio','buy_money','use_day', 'end_date'])
        total.to_pickle(os.path.join(BASE_DIR, 'results', 'sub_ris.pkl'))
        if latest_result:
            pd.DataFrame(latest_result, columns=['symbol', 'date']).to_pickle(
                os.path.join(BASE_DIR, 'results', 'sub_rsi_latest.pkl'))
        print(total)
        print(total.profit.mean())

    def run(self,symbol,item,min_rsi):
        item.dropna(subset=['reduce'], inplace=True)
        item.set_index('date', inplace=True, drop=False)
        item['max_diff_60'] = item.reduce.rolling(window=60).apply(self.get_min_difference, args=(1,))
        item['max_diff_45'] = item.reduce.rolling(window=45).apply(self.get_min_difference, args=(1,))
        item['max_diff_30'] = item.reduce.rolling(window=30).apply(self.get_min_difference, args=(1,))
        item['is_min_rsi'] = item.rsi_6.rolling(window=min_rsi_window).apply(self.get_min_rsi)
        item['max_reduce'] = (item['max_diff_60'] + item['max_diff_45'] + item['max_diff_30']) >= 2
        item['slope'] = self.get_slope(item.rsi_6)
        profits = []
        latest_list = []
        condition = (item.max_reduce)
        if min_rsi:
            condition &= (item.is_min_rsi)
        for date in item.loc[condition,'date']:
            # if self.get_slope(item.loc[item.date<=date,'rsi_6']) >= min_slope:
            #     continue
            if item.loc[item.date==date,'rsi_6'].iloc[0] > min_rsi_value:
                continue
            if date == item.date.max():
                latest_list.append([symbol,date])
            if profits and date <= profits[-1][-1]:
                continue
            try:
                profits.append(self.get_profit(date,item,symbol))
            except IndexError:
                self.money = self.local_money
                continue

        return profits,latest_list


if __name__ == '__main__':
    from common.utils import load_df, BASE_DIR
    stock = Trading.load_stock()
    t = tqdm(range(len(stock.symbol.unique())), ncols=80)
    pool = Pool(4)
    jobs = []
    result = []
    latest_result = []
    min_rsi = False
    for symbol,item in stock.groupby('symbol'):
    # for symbol, item in stock.query("symbol=='300393.SZ'").groupby('symbol'):
        obj = Trading(money=90000)
        jobs.append(pool.apply_async(obj.run,args=(symbol,item,min_rsi)))
    for job in jobs:
        winner,latest_list = job.get()
        if winner:
            result.extend(winner)
            latest_result.extend(latest_list)
        t.update(1)
    Trading.save(result,latest_result)
