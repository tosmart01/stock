import tushare as ts
import baostock as bs
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from os.path import join, dirname
import os
import pandas as pd
from tqdm import tqdm
from common.logger import logger
from glob import glob
from common.utils import collection_error


class Astock_download:

    def __init__(self, start=None, end=None, timeToMarket=20210101,):
        self.start = start
        self.end = end
        self.lock = Lock()
        self.symbol_list = self.get_stock_code(timeToMarket)
        self.pool = ThreadPoolExecutor(max_workers=15)
        self.stock = pd.DataFrame(data=None)

    def get_stock_code(self, timeToMarket):
        # symbol = ts.get_stock_basics()
        bs.login()
        pro = ts.pro_api()
        symbol = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        symbol =  symbol.query(f"list_date<'{timeToMarket}'")
        symbol.ts_code = symbol.ts_code.apply(lambda x: x.split('.')[1].lower() + '.' + x.split('.')[0])
        symbol.set_index('symbol',inplace=True)
        print(symbol)

        return symbol

    def get_pe(self,code,df):
        # try:
        #     bs.login()
        #     # en = str((pd.to_datetime(self.end) - timedelta(days=10)).date())
        #     rs = bs.query_history_k_data_plus(code, "date,code,peTTM", start_date=self.start,
        #                                       end_date=self.end, frequency="d", adjustflag="3")
        #     pe = rs.get_data()
        #     pe.code = pe.code.apply(lambda x: x.split('.')[1])
        #     pe.rename(columns={'code': 'symbol','peTTM':'pe'}, inplace=True)
        #     df['pe'] = df.merge(pe,on=['symbol','date']).pe
        #     df['pe'].fillna(method='ffill',inplace=True)
        # except Exception as e:
        #     print(e)
        # else:
        df['pe'] = 60
        return df

    @collection_error()
    def down_load(self, code, value, tq):

        # print(code,datetime.now())
        df = ts.get_hist_data(code, start=self.start, end=self.end)
        if df is not None and not df.empty:
            df.reset_index('date',inplace=True)
            df['symbol'], df['name'], df['industry']  = code, value.name, value.industry
            df = self.get_pe(value.ts_code,df)
            with self.lock:
                self.stock = self.stock.append(df)  # 一次性获取全部日k线数据
        tq.update(1)

    def get_stock_close(self):
        today = datetime.now().date()
        self.end = self.end if self.end else str(today)
        self.start = self.start if self.start else (today - timedelta(days=360)).strftime('%Y-%m-%d')
        tq = tqdm(range(len(self.symbol_list)), desc='download: ',ncols=80)
        jobs = []
        for value in self.symbol_list.itertuples():
            jobs.append(self.pool.submit(self.down_load, value.Index, value, tq))
        for i in jobs:
            try:
                i.result(10)
            except Exception as e:
                print(e)
                tq.update(1)
        self.stock.reset_index(inplace=True)



def main(update=True,start='2019-03-01'):
    logger.info('stock 数据更新')
    history = glob(join(dirname(__file__),'history_data','*.pkl'))
    for file in sorted(history,reverse=True)[10:]:
        os.remove(file)
    start = start
    save_path = join(dirname(__file__), 'history_data', f"stock_{datetime.today().date()}.pkl")
    if history:
        history = sorted([(i,os.stat(i).st_mtime) for i in history],key=lambda x:x[1])[-1][0]
        df = pd.read_pickle(history)
        if not update:
            return df
        else:
            max_date = df.date.max()
            start = (pd.to_datetime(max_date) - timedelta(days=5)).date()
            fs = Astock_download(start=str(start))
            fs.get_stock_close()

            df = df.append(fs.stock)
            df = df.drop_duplicates(['symbol', 'date'])
            df.to_pickle(save_path)
            logger.info('stock 数据更新完成')
            return df
    else:
        fs = Astock_download(start=str(start))
        fs.get_stock_close()
        fs.stock.to_pickle(save_path)
        logger.info('stock 数据更新完成')
        return fs.stock




if __name__ == '__main__':
    main()

