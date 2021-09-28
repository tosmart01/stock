# -- coding: utf-8 --
# @Time : 2021/9/19 17:09
# @Author : zhuo.wang
# @File : stock_god.py
from os.path import dirname, join

import pandas as pd
from glob import glob
from datetime import datetime
from tqdm import tqdm
import numpy as np

pd.set_option("display.max_rows", 300 * 5)


def gen_total_df(df_slope,date_uniue):
    container = {}
    for index, row in df_slope.iterrows():
        date_range = pd.date_range(
            date_uniue[date_uniue.index(row.end_date) - interval[-1]], row.end_date
        )
        if len(date_range) == 0:
            continue
        exists = False
        for date in date_range:
            record = container.get(date, {})
            if record:
                exists = True
                if row.slope > record["slope"]:
                    del container[date]
                    container[row.end_date] = row.to_dict()
        if not exists:
            container[date] = row.to_dict()
    if container:
        total = pd.DataFrame(list(container.values()))
        total["gains"] = total["close_price"] - total["start_price"]
        for name, g in golden_cut:
            total[name] = total["gains"] * g + total["start_price"]
        return total
    return pd.DataFrame()

def get_buy_data(name,data,row):
    # 寻找最接近黄金点买入日期
    if name == "G0":
        buy_data = (
            data.loc[
                (data.low <= row[name])
                & (data.high * 1.02 >= row[name])
                & (data.date > row.end_date)
                ]
                .sort_values("date")
                .iloc[0:1]
        )
    else:
        buy_data = (
            data.loc[
                (data.low <= row[name])
                & (data.high >= row[name])
                & (data.date > row.end_date)
                ]
                .sort_values("date")
                .iloc[0:1]
        )
    return buy_data


def get_point(data,buy_index,god_name,buy_price):
    point_win = []
    point_loss = []
    for name,value in win_point.items():
        is_win =  ((data.loc[buy_index + 1:buy_index + sell_interval[god_name]].high / buy_price) >= value).any()
        point_win.append(is_win)

    for name,value in loss_point.items():
        is_win =  ((data.loc[buy_index + 1:buy_index + sell_interval[god_name]].high / buy_price) <= value).any()
        point_loss.append(is_win)
    return point_win,point_loss

def shadow_line(data:pd.DataFrame) -> bool:
    '''
    判断买入日期前两天内 有没有下影线
    :param row:
    :return:
    '''
    is_shadow = False
    for index,row in data.iterrows():
        try:
            if row.p_change < 0:
                is_shadow = (row.close - row.low) / (row.open - row.close)>= 0.5
            else:
                is_shadow = (row.open - row.low) / (row.close - row.open) >= 0.5
        except ZeroDivisionError:
            continue
        if is_shadow:
            return is_shadow

    return is_shadow


def gen_profits(total, data):
    profits = []
    for index, row in total.iterrows():
        for name, god in golden_cut:
            buy_data = get_buy_data(name,data,row)
            # 买入为空，则判断是否在 区间中，在取最大值做买入日期
            if buy_data.empty:
                if (data.date.max() - row.end_date).days < interval[-1] + 10:
                    buy_data = data.loc[data.date == data.date.max()].iloc[0:1]
                else:
                    continue
            buy_index = buy_data.index[0]
            # 如果 连续 5天跌停，跳过该股票
            if (
                data.loc[data.date.between(row.end_date, buy_data.date.iloc[0])]
                .query("-11<=p_change<=-9")
                .__len__()
                >= 5
            ):
                continue
            # 控制买入日期在 窗口期内
            if (
                buy_index - data.loc[data.date == row.end_date].index[0]
            ) > god_day_range[name]:
                continue
            # 如果 买入价格 不在股价内跳过
            buy_price = row[name]  # buy_data.close.iloc[0]
            # 股价前5天内是否出现超跌
            is_oversold = False
            if (data.loc[buy_index - 5:buy_index].p_change.sum() <= -8):
                is_oversold = True

            is_shadow = shadow_line(data.loc[buy_index:buy_index])
            # 如果是最近发生的买点，则记录nan
            if buy_index + sell_interval[name] > data.index.max():
                sell_date = today
                sell_price = np.nan
                point_win = [False] * len(win_point)
                point_loss = [False] * len(loss_point)
            else:
                # 使用卖出日期内最高价
                max_price_id = data.loc[
                    buy_index + 1 : buy_index + sell_interval[name], "high"
                ].idxmax()
                sell_price = data.loc[max_price_id, "high"]
                sell_date = data.loc[max_price_id, "date"]
                point_win,point_loss = get_point(data,buy_index,name,buy_price)
            profits.append(
                [
                    (sell_price - buy_price) / buy_price,
                    name,
                    buy_data.date.iloc[0],
                    row.slope,
                    sell_date,
                    sell_interval[name],
                    buy_price,
                    sell_price,
                    is_shadow,
                    is_oversold,
                    row.end_date,
                    row.start_date,
                    row.window,
                    row.G6,
                    row.G5,
                    row.G3,
                    row.G1,
                    row.G0,
                    *point_win,
                    *point_loss,
                    row["index"],
                ]
            )
    result = pd.DataFrame(
        profits,
        columns=[
            "gains",
            "_type",
            "buy_date",
            "slope",
            "sell_date",
            "sell_interval",
            "buy_price",
            "sell_price",
            "is_shadow",
            'is_oversold',
            "end_date",
            "start_date",
            "window",
            "G6",
            "G5",
            "G3",
            "G1",
            "G0",
            *win_point.keys(),
            *loss_point.keys(),
            "index",
        ],
    )
    return result


def gen_slope_df(data):
    res = []
    for window in interval:
        open_value = data.open.shift(window - 1)
        slope = pd.DataFrame((data.high - open_value) / open_value, columns=["slope"])
        slope["window"] = window
        res.append(slope)

    df_slope = pd.concat(res).reset_index().query(f"slope >= {min_slope}")
    df_slope["end_date"] = df_slope["index"].apply(lambda x: date_map.get(x))
    df_slope["start_date"] = (df_slope["index"] - df_slope.window + 1).apply(
        lambda x: date_map.get(x)
    )
    df_slope = df_slope.dropna(subset=["slope"]).sort_values(["end_date"])
    df_slope["close_price"] = df_slope["index"].apply(lambda x: data.loc[x, "high"])
    df_slope["start_price"] = df_slope["start_date"].apply(
        lambda x: data.loc[data.date == x, "open"].iloc[0]
    )
    return df_slope


def load_df(file_path):
    df = pd.read_pickle(file_path)
    df["date"] = df.date.astype("datetime64")
    df.sort_values(["symbol", "date"], inplace=True, ignore_index=True)

    df["turnover"] = df.volume * df.close * 100
    return df

def run():
    res = []
    max_date = df.date.max()
    t = tqdm(
        range(
            df.loc[(~df.name.str.contains("ST"))
                   #& (~df.symbol.str.startswith("3"))
                   ]
            .name.unique()
            .__len__()
        ),
        ncols=70,
    )
    for stock_name, data in df.loc[
        (~df.name.str.contains("ST"))
        #& (~df.symbol.str.startswith("3"))
    ].groupby("name"):
        t.update(1)
        date_uniue = list(data.date.unique())
        df_slope = gen_slope_df(data)
        slope_max_date = df_slope.end_date.max()
        if df_slope.empty or slope_max_date >= max_date:
            continue
        total = gen_total_df(df_slope, date_uniue)
        if total.empty:
            continue
        result = gen_profits(total, data)
        result["stock_name"] = stock_name
        res.append(result)

    cur_close = df.sort_values(['date','name'],ascending=False).drop_duplicates(['name'],keep='first').rename(columns={'name':'stock_name'})[['stock_name','symbol','close']]
    cc = pd.concat(res).merge(cur_close, how='left')
    cc = cc.set_index('stock_name').reset_index().drop('index', axis=1)
    cc['gains_sign'] = cc['gains'].apply(lambda x: x > 0.0)
    return cc

if __name__ == "__main__":
    base_dir = dirname(dirname(__file__))
    file_list = glob(join(base_dir,'history_data','*.pkl'))
    file_path = sorted(file_list)[-1]
    print(file_path)
    df = load_df(file_path)

    date_map = dict(zip(df.index, df.date))
    today = datetime.now()
    interval = range(12, 45)
    min_slope = 0.86
    golden_cut = [("G6", 0.618), ("G5", 0.5), ("G3", 0.382), ("G1", 0.191), ("G0", 0)]
    sell_interval = {"G6": 2, "G5": 3, "G3": 4, "G1": 12, "G0": 20}
    win_point = {f"point_{i}":round(1 + i/100,2) for i in range(1,23)}
    loss_point = {f"loss_point_{i}": 2 - round(1 + i / 100, 2) for i in range(1, 23)}
    god_day_range = {
        "G6": 8,
        "G5": interval[-1] * 0.5,
        "G3": interval[-1] * 0.8,
        "G1": interval[-1] * 1.2,
        "G0": interval[-1] * 2,
    }

    res = run()

    res.to_pickle(join(base_dir,'results','god.pkl'))