# -- coding: utf-8 --
# @Time : 2021/10/6 15:31
# @Author : zhuo.wang
# @File : backtest.py
import pandas as pd
from glob import glob
from os.path import abspath,dirname,join
import tushare as ts
from copy import deepcopy

def load_df():
    base_dir = dirname(dirname(abspath(__file__)))
    file_list = glob(join(base_dir,'history_data','*.pkl'))
    file_path = sorted(file_list)[-1]
    df = pd.read_pickle(file_path)
    df['date'] = df.date.astype('datetime64')
    df.sort_values(['symbol', 'date'], inplace=True, ignore_index=True)
    gains = pd.read_pickle(join(base_dir,'results','god.pkl'))
    gains = gains.loc[gains.gains.notna()]
    return df,gains

def get_sell_cw(item,cw):
    sell_cw = item['sell_cw']
    if item['sell_cw'] > cw:
        sell_cw = cw
    return sell_cw

def sub_profits(loss_point,loss_A_copy,loss_B_copy,profits,cw,bj,sell_count):
    if loss_point <= loss_A_copy['point'] and not loss_A_copy.get("repeat"):
        sell_cw = get_sell_cw(loss_A_copy,cw)
        profits += bj * sell_cw * loss_point
        cw -= sell_cw
        loss_A_copy['repeat'] = True
        sell_count += 1
    elif loss_point <= loss_B_copy['point'] and not loss_B_copy.get("repeat"):
        sell_cw = get_sell_cw(loss_B_copy,cw)
        profits += bj * sell_cw * loss_point
        cw -= sell_cw
        loss_B_copy['repeat'] = True
        sell_count += 1
    return cw,profits,sell_count

def get_profits(value):
    profits = 0
    sell_count = 0
    for ix,row in value.iterrows():
        # 仓位
        cw = 1
        point_pool = deepcopy(point_config)
        loss_A_copy = deepcopy(loss_A)
        loss_B_copy = deepcopy(loss_B)
        sell_day = sell_interval[row._type]
        prices = df.loc[(df.name==row.stock_name)&(df.date>=row.buy_date)].sort_values('date').iloc[:sell_day]
        for price in prices.itertuples():
            gains_point = (price.high / row.buy_price) - 1
            loss_point = (price.low / row.buy_price) - 1
#             print("loss_point is :",loss_point)
            if cw < -0.0001 or cw == 0:
                end_date = price.date
                break
            # 如果仓位小于0 结束操作
            for point_item in point_pool:
                if point_item['point'] <= gains_point  and not point_item.get("repeat"):
                    # 如果卖出仓位大于剩余仓位，则全部卖出
                    sell_cw = get_sell_cw(point_item,cw)
                    profits += bj * gains_point * sell_cw
                    point_item['repeat'] = True
                    cw -= sell_cw
                    end_date = price.date
                    sell_count += 1
#                     print(sell_cw,cw,gains_point)
            cw,profits,sell_count = sub_profits(loss_point,loss_A_copy,loss_B_copy,profits,cw,bj,sell_count)

        if cw > 0:
            last_point = (prices.iloc[-1].close / row.buy_price) - 1
            profits += bj * last_point * cw
            cw = 0
            end_date = prices.iloc[-1].date
            sell_count += 1
        use_day = (set(pd.date_range(row.buy_date,end_date)) & day).__len__()
    return profits,use_day,sell_count

if __name__ == '__main__':
    df,gains = load_df()
    day = set(df.date)
    point_config = [
        {"name": "point_30%", "point": 0.04, 'sell_cw': 0.3},
        {"name": "point_30%", "point": 0.07, 'sell_cw': 0.3},
        {"name": "point_20%", "point": 0.10, 'sell_cw': 0.2},
        {"name": "point_20%", "point": 0.13, 'sell_cw': 0.2},
    ]
    loss_A = {
        "point": -0.03,
        "sell_cw": 0.4
    }
    loss_B = {
        "point": -0.05,
        "sell_cw": 0.6
    }
    sell_interval = {"G6": 2, "G5": 3, "G3": 4, "G1": 12, "G0": 20}
    # 本金5万
    res = []
    bj = 50000
    for rs in range(12):
        for index, value in gains.query("_type == 'G1' and  stock_name=='宝兰德'").sample(n=1).groupby(
                ['stock_name', 'buy_date']):
            profits, use_day, sell_count = get_profits(value)
            bj += profits
            res.append([index, profits, use_day])
            print(index, profits, use_day, bj, sell_count)
