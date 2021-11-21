# -- coding: utf-8 --
# @Time : 2021/10/6 15:31
# @Author : zhuo.wang
# @File : backtest.py
import pandas as pd
from glob import glob
from os.path import abspath, dirname, join
import tushare as ts
from copy import deepcopy


def load_df():
    base_dir = dirname(dirname(abspath(__file__)))
    file_list = glob(join(base_dir, 'history_data', '*.pkl'))
    file_path = sorted(file_list)[-1]
    df = pd.read_pickle(file_path)
    df['date'] = df.date.astype('datetime64')
    df.sort_values(['symbol', 'date'], inplace=True, ignore_index=True)
    gains = pd.read_pickle(join(base_dir, 'results', 'god.pkl'))
    gains = gains.loc[gains.gains.notna()]
    return df, gains


def get_sell_cw(item, cw):
    sell_cw = item['sell_cw']
    if item['sell_cw'] > cw:
        sell_cw = cw
    return sell_cw


def sub_profits(loss_point, loss_A_copy, loss_B_copy, profits, cw, benjin, sell_count):
    if cw < -0.0001 or cw == 0:
        return cw, profits, sell_count,benjin
    if loss_point <= loss_A_copy['point'] and not loss_A_copy.get("repeat"):
        sell_cw = get_sell_cw(loss_A_copy, cw)
        profits += benjin*cw * sell_cw * loss_A_copy['point']
        benjin += profits
        cw -= sell_cw
        loss_A_copy['repeat'] = True
        sell_count += 1
    elif loss_point <= loss_B_copy['point'] and not loss_B_copy.get("repeat"):
        sell_cw = get_sell_cw(loss_B_copy, cw)
        profits += benjin*cw * sell_cw * loss_B_copy['point']
        benjin += profits
        cw -= sell_cw
        loss_B_copy['repeat'] = True
        sell_count += 1
    return cw, profits, sell_count,benjin


def is_rising_trend(prices, row):
    '''
    判断买入后两天内是否为上升趋势，如果是，加仓
    :param prices:
    :param row:
    :return:
    '''
    day1 = prices.high.iloc[0]
    day2 = prices.high.iloc[1]
    w1 = day1 > row.buy_price
    w2 = day2 > row.buy_price
    return all([w1, w2])

def avg_point(prices,buy_price):
    res = []
    for price,index in zip(prices.close,day_point_val):
        if (price / buy_price - 1) >= index:
            res.append(True)
        else:
            res.append(False)
    return res

def trading(prices, row, benjin, sell_count, profits, cw, end_date):
    # 判断是否加仓
    point_pool = deepcopy(point_config)
    loss_A_copy = deepcopy(loss_A)
    loss_B_copy = deepcopy(loss_B)
    point_list = avg_point(prices,row.buy_price)
    for ix, price in enumerate(prices.itertuples(), start=1):

        loss_point = (price.low / row.buy_price) - 1
        gains_point = (price.high / row.buy_price) - 1

        if price.high >= row.G3 * 1.05:
            end_date = price.date
            break
        if cw < -0.0001 or cw == 0:
            end_date = price.date
            break

        # 如果仓位小于0 结束操作
        for point_item in point_pool:
            if point_item['point'] <= gains_point and not point_item.get("repeat") and not (cw < -0.0001 or cw == 0):
                # 如果卖出仓位大于剩余仓位，则全部卖出
                sell_cw = get_sell_cw(point_item, cw)
                profits += benjin*cw * point_item['point'] * sell_cw
                benjin += profits
                point_item['repeat'] = True
                cw -= sell_cw
                end_date = price.date
                sell_count += 1
        cw, profits, sell_count,benjin = sub_profits(loss_point, loss_A_copy, loss_B_copy, profits, cw, benjin, sell_count)

    return cw, sell_count, profits, end_date,point_list


def get_profits(value, benjin=None):
    profits = 0
    sell_count = 0
    use_day = 0
    for ix, row in value.iterrows():
        # 仓位
        cw = 1
        sell_day = sell_interval[row._type]
        prices = df.loc[(df.name == row.stock_name) & (df.date > row.buy_date)].sort_values('date').iloc[:sell_day]
        indexs = prices.index.tolist()
        max_price_id = prices.close.idxmax()
        max_date = indexs.index(max_price_id)
        max_ratio = (prices.loc[max_price_id,'close'] / row.buy_price - 1)
        end_date = prices.date.max()
        cw, sell_count, profits, end_date,point_list = trading(prices, row, benjin, sell_count, profits, cw, end_date)
        # if profits / benjin > 0.06:
        #     prices = df.loc[(df.name == row.stock_name) & (df.date > row.buy_date)].sort_values('date').iloc[
        #              sell_day:int(sell_day * 1.6)]
        #     cw, sell_count, profits, end_date = trading(prices, row, benjin, sell_count, profits, cw,
        #                                                          end_date, )
        if cw > 0:
            last_point = (prices.loc[prices.date==end_date,'close'].iloc[0] / row.buy_price) - 1
            profits += (benjin * cw + profits) * last_point * cw
            cw = 0
            end_date = prices.iloc[-1].date
            sell_count += 1
        use_day = (set(pd.date_range(row.buy_date, end_date)) & day).__len__()

    return profits, use_day, sell_count, benjin,max_date,max_ratio,point_list


def shadow_line(data:pd.DataFrame) -> bool:
    '''
    判断买入日期前两天内 有没有下影线
    :param row:
    :return:
    '''
    is_shadow = False
    for index,row in data.iterrows():
        p_change = row.close > row.open
        try:
            if not p_change:
                is_shadow = (row.close - row.low) / (row.high - row.low)>= 0.4
            else:
                is_shadow = (row.open - row.low) / (row.high - row.low) >= 0.4
        except ZeroDivisionError:
            continue
        if is_shadow:
            return is_shadow
    return is_shadow

if __name__ == '__main__':
    df, gains = load_df()
    day = set(df.date)

    point_config = [
        {"name": "point_A", "point": 0.06, 'sell_cw': 0.0},
        {"name": "point_B", "point": 0.15, 'sell_cw': 0.0},
        {"name": "point_C", "point": 0.15, 'sell_cw': 0.0},
        {"name": "point_D", "point": 0.15, 'sell_cw': 0.0},
    ]
    loss_A = {
        "point": -0.03,
        "sell_cw": 0.3
    }
    loss_B = {
        "point": -0.05,
        "sell_cw": 0.7
    }
    sell_interval = {"G6": 2, "G5": 3, "G3": 3, "G1": 7, "G0": 20}
    day_point_col = [f"day_{i}" for i in range(1,sell_interval['G1']+1)]
    day_point_val = [i / 100 for i in range(1,sell_interval['G1']+1)]
    # 本金5万
    res = []
    for rs in range(12):
        bj = 40000
        for index, value in gains.query("_type == 'G1' and  stock_name=='东方中科'").sample(n=1).groupby(
                ['stock_name', 'buy_date']):
            profits, use_day, sell_count, benjin,point_list = get_profits(value, bj)
            bj += profits
            res.append([index, profits, benjin, use_day])
            print(index, profits, use_day, benjin, sell_count)
