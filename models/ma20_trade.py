# -- coding: utf-8 --
# @Time : 2022/1/7 0:19
# @Author : zhuo.wang
# @File : ma20_trade.py
from datetime import datetime
import os

import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

from common.utils import load_df,BASE_DIR
df = load_df()
date_map = dict(zip(df.index, df.date))
interval = range(45, 200)
min_slope = 0.6

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
#     df_slope["close_price"] = df_slope["index"].apply(lambda x: data.loc[x, "high"])
#     df_slope["start_price"] = df_slope["start_date"].apply(
#         lambda x: data.loc[data.date == x, "open"].iloc[0]
#     )
    return df_slope

def gen_total_df(df_slope,date_uniue):
    container = {}
    for row in df_slope.itertuples():
        start_index = date_uniue.index(row.end_date) - interval[-1]
        end_index = date_uniue.index(row.end_date)
        if end_index < start_index and start_index <0:
            continue
        exists = False
        while end_index > start_index:
            start_date = date_uniue[start_index]
            record = container.get(start_date, {})
            if record:
                exists = True
                if row.slope > record["slope"]:
                    del container[start_date]
                    container[row.end_date] = row._asdict()
            start_index += 1
        if not exists:
            container[row.end_date] = row._asdict()
    if container:
        total = pd.DataFrame(list(container.values()))
        return total
    return pd.DataFrame()

def main(name,value):
    slope = gen_slope_df(value)
    unique_date = value.date.drop_duplicates().tolist()
    result = gen_total_df(slope, unique_date)
    result['symbol'] = name
    result['stock_name'] = value.name.iloc[0]
    return result

if __name__ == '__main__':
    res = []
    t = tqdm(range(len(df.symbol.unique())))
    pool = Pool(3)
    jobs = []
    for name, value in df.groupby('ts_code'):
        jobs.append(pool.apply_async(main,args=(name,value)))

    for job in jobs:
        res.append(job.get())
        t.update(1)
    total = pd.concat(res,ignore_index=True)
    total.to_pickle(os.path.join(BASE_DIR,'results',f'{datetime.today().date()}_ma20.pkl'))