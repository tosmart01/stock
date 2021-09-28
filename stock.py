import warnings
from glob import glob

warnings.filterwarnings("ignore")
import os
import sys
import time
import download_data
from datetime import datetime, timedelta
from os.path import dirname, abspath, join
from tqdm.autonotebook import tqdm
from models.scheduler import Average, EMAverage
import pandas as pd
from multiprocessing import Pool
from common.logger import logger
from common.utils import write_db, BASE_DIR
from sqlalchemy import create_engine, VARCHAR, INTEGER
import pymysql

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Stock:

    def __init__(self, max_worker=6, col=[], df=None, bf_day=6, var=0.02,
                 line_count=2, model='get_ma_B', progress=True, model_class=Average):

        self.result = []
        self.origin = []
        self.df = df
        self.col = col
        self.var = var  # 方差系数
        self.line_count = line_count  # 均线增长数量
        self.pool = Pool(max_worker)
        self.bf_day = bf_day
        self.model = model
        self.progress = progress
        self.model_class = model_class
        if progress:
            self.tq = tqdm(range(len(set(self.df.symbol))), ncols=80)

    def run(self):
        df = self.df
        st = time.time()
        logger.info(f"{self.col[-1]} {self.model} 计算开始,股票数:{len(set(df.symbol))}")

        model = self.model_class(self.line_count, self.var, self.col,  self.bf_day)
        jobs = []
        for index, value in df.groupby(df.symbol):
            # self.result.append(model.execute(index, value))
            jobs.append(self.pool.apply_async(model.execute, args=(index, value)))
        self.pool.close()
        for i in jobs:
            res, origin = i.get()
            self.result.append(res)
            self.origin.append(origin)
            if self.progress:
                self.tq.update(1)

        logger.info(f"{self.col[-1]} 计算完成,用时:{(time.time() - st):.2f}s")
        return self.result, self.origin
    # print(time.time() - ss)

    def __del__(self):
        del self.origin,self.result


def rank_mul(diff, stock):
    now = datetime.today()
    origin = get_diff_origin(diff)
    now = origin[origin.date < now].date.iloc[-diff]
    pord_dict = {}
    for symbol, df in stock.groupby(['symbol']):
        date_list = {}
        origin_df = origin[origin.symbol == symbol].sort_values('date').set_index('date', drop=False)
        min_date = origin_df.index.min() + timedelta(days=10)
        for date in df.jx_date:
            month = date.strftime('%Y%m')
            if month not in date_list and date < now and date >= min_date:
                date_list[month] = date
        date_list = [date_list[i] for i in date_list]
        success = origin_df[origin_df.date.isin(date_list)].af_colse_15 > diff / 100 * 0.6
        pord = success.sum() / len(success)
        pord_dict[symbol] = pord
    return pord_dict


def rank(stock, var, diff=None):
    pool = Pool(2)
    stock.jx_date = stock.jx_date.astype('datetime64')
    stock = stock.loc[(stock['var'] <= var) & (stock.pe <= 68) & (stock.jx_counts >= 2)]
    jobs = [(_diff, pool.apply_async(rank_mul, args=(_diff, stock))) for _diff in diff]
    pool.close()
    for _diff, job in jobs:
        pord_dict = job.get()
        stock[f'pord_{_diff}'] = stock.symbol.apply(lambda x: pord_dict[x])
    return stock


def get_diff_origin(diff=12):
    history = glob(join(dirname(__file__), 'history_data', '*.pkl'))
    history = sorted([(i, os.stat(i).st_mtime) for i in history], key=lambda x: x[1])[-1][0]
    origin = pd.read_pickle(history)
    origin.date = origin.date.astype('datetime64')
    origin = origin.sort_values(['symbol', 'date']).reset_index(drop=True)
    origin['af_colse_15'] = -origin.groupby('symbol').apply(lambda x: x['close'].diff(-diff) / x['close']). \
        reset_index().sort_values('level_1')['close']
    return origin


def save_result(db_result):
    con = pymysql.connect(user='root', port=3306, host='www.chaoyue.red', password='123456', db='bbs_1')
    db = con.cursor()
    db.execute('truncate table stock')
    db.execute('truncate table stock_base')
    con.commit()
    for models, stock, origin,col in db_result:
        logger.info(f"{models + '_' + col[-1]} 开始保存csv")
        stock.to_csv(join(BASE_DIR, 'results', f"{models + '_' +col[-1]}_stock.csv"), index=False)
        origin.to_csv(join(BASE_DIR, 'results', f"{models + '_' + col[-1]}_stock_base.csv"), index=False)

        logger.info(f"{models + '_' + col[-1]} 开始入库")
        stock.drop('index',axis=1,inplace=True)
        origin.drop('index',axis=1,inplace=True)
        write_db(stock, con=con, table_name='stock',chunksize=3000)
        write_db(origin, con=con, table_name='stock_base', chunksize=3000)
        logger.info(f"{models + '_' + col[-1]} 入库结束")

def init():

    window = (datetime.today() - timedelta(days=365 * 2)).strftime('%Y-%m-%d')
    update = True if len(sys.argv) > 1 else False
    df = download_data.main(update=update, start=window)
    pkl_list = sorted([(i, os.stat(i).st_mtime) for i in glob(join(BASE_DIR, 'history_data', '*.pkl'))],
                      key=lambda x: x[1])
    for pkl, stat in pkl_list[:-2]:
        logger.info(f'清除历史数据:{pkl}')
        os.remove(pkl)
    df.sort_values(['symbol', 'date'], inplace=True)
    df.set_index('date', inplace=True, drop=False)
    df.date = df.date.astype('datetime64')
    df.rename(columns={'date': 't_date'}, inplace=True)
    df.index = pd.to_datetime(df.index)
    # df = df.query("name=='牧原股份' and date>'2018-04-01'")
    df = df.query(f"date>='{window}'")
    return df

if __name__ == '__main__':
    try:
        st = time.time()
        # if datetime.weekday(datetime.today()) >= 5:
        #     logger.info('周末不执行')
        #     os._exit(0)
        ma60 = ['ma5', 'ma10', 'ma20', 'ma30', 'ma60']
        ma30 = ['ma5', 'ma10', 'ma20', 'ma30', ]
        jobs = {
            "1": ('get_ma_A', ma60, 0.028),
            "3": ('get_ma_A', ma30, 0.014),
        }
        db_result,df = [],init()
        for models, col, var in jobs.values():
            res, origin = Stock(col=col,
                                line_count=2,
                                var=var,
                                max_worker=5,
                                progress=True,
                                bf_day=6,
                                df=df,
                                model_class=Average
                                ).run()

            stock = pd.concat(res)
            origin = pd.concat(origin)
            stock.rename(columns={"name": 'stock_name', "open": 'open_price', "close": "close_price"}, inplace=True)
            origin.rename(columns={"name": 'stock_name', "open": 'open_price', "close": "close_price"}, inplace=True)
            stock['sw_class'] = models + '_' + col[-1]
            origin['sw_class'] = models + '_' + col[-1]
            # 计算胜率
            logger.info(f"{models + '_' + col[-1]} 开始计算胜率")
            stock = rank(stock, var, diff=[5, 10, 15, 30])
            logger.info(f"{models + '_' + col[-1]} 胜率计算结束")
            stock = stock.round(4)
            db_result.append((models, stock, origin,col))

        save_result(db_result)

        logger.info(f'stock 计算结束,用时:{(time.time() - st)/60:.2f}m')
        os._exit(0)

    except Exception as e:
        logger.exception(e)