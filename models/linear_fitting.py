import time
from datetime import datetime,timedelta
from os.path import join,dirname,abspath
base_dir = dirname(dirname(abspath(__file__)))

import warnings,sys
from itertools import chain
warnings.filterwarnings('ignore')
sys.path.append(base_dir)

from sympy import expand,Symbol,diff,solve
from glob import glob
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from common.logger import logger
from common.utils import write_db
from scipy import signal   #滤波等

class fit:
    def __init__(self,bf_day=15):
        self.base_dir = base_dir
        self.bf_day = bf_day


    def load_data(self):
        path = join(self.base_dir,'history_data')
        pkl = sorted(glob(join(path,'*.pkl')),reverse=True)[0]
        print(pkl)
        df =  pd.read_pickle(pkl)
        df['close_day'] = df.close * df.volume * 100 / 10000000
        symbol_list = pd.DataFrame(df.query(f"date>='{(datetime.today() - timedelta(7)).date()}'").
                                   groupby('symbol')['close_day'].mean()>=30)
        df = df.loc[df.symbol.isin(symbol_list.loc[symbol_list.close_day>0].index)]
        print(df)
        df['date'] = df['date'].astype('datetime64')
        df = df.loc[df['date'] >= (datetime.today() - timedelta(200)).strftime('%Y-%m-%d')]
        df.sort_values(['symbol', 'date'], inplace=True)
        df.set_index('name', drop=False, inplace=True)
        return df

    def get_func(self,x, y, deg=2):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            _f = np.polyfit(np.array(x), np.array(y), deg)
            return np.poly1d(_f)

    def find_nearest(self,array, value):
        '''
        获取数组中 近似值
        '''
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]



    def conclusion(self,stock,yici_func, func):
        '''
        yici_func: 一次方程组
        func: em5  方程组
        根据 一次方程组和三次方程组求 交集
        返回 (symbol,符合日期)
        '''
        results, _range = [], []
        result = list(filter(lambda x: func(x) >= yici_func(x), stock.index))
        for index, v in enumerate(result):
            if v + 1 not in result:
                _range.append(index)
        for index, v in enumerate(_range):
            if index == 0:
                results.append(result[:v + 1])
            else:
                _index = _range[_range.index(v) - 1] + 1
                results.append(result[_index:v + 1])
        date_list = []
        for i in results:
            slope = np.polyfit(i, func(i), 1)[0]
            if slope > 0:
                date_list.append(i)
        return list(filter(lambda x: np.polyfit(x, stock['close_day'][x], 1)[0], date_list[-1:]))


    def get_slope(self,stock, func2, ):
        '''
        根据em5 极值点，
        在根据极值点 求 3次方程
        三次方程求导 得到 单调区间
        根据单调区间  求得 一次方程组，
        return  一次方程组
        '''
        # 极值点
        peak = list(signal.find_peaks(stock['em5'], distance=15)[0])  # distance表极大值点的距离至少大于等于10个水平单位
        func = self.get_func(peak, [stock['em5'][i] for i in peak], 3) #三次方程组

        #求导
        x = Symbol('x')
        f_exp = expand(func(x))
        dify = diff(f_exp)
        ep = solve(dify, x)

        _range = list(map(round, ep))
        v1, v2 = self.find_nearest(peak, _range[0]), self.find_nearest(peak, _range[1])

        #区间判断
        if v2 >= stock.index[-self.bf_day]:
            #还原数组
            new_peak = peak[peak.index(v1):peak.index(v2) + 1]

            #  单调区间
            if dify.evalf(subs={'x': _range[1]}) < dify.evalf(subs={'x': _range[1] + 1}):
                f2 = self.get_func(new_peak, [func2(i) for i in new_peak], 1)
                return new_peak, f2(new_peak), f2


    def execute(self,symbol, stock,):
        stock.reset_index(drop=True, inplace=True)
        stock['m5'] = stock['close'].rolling(5, min_periods=1).mean()
        stock['em5'] = stock['close'].ewm(span=5).mean()
        # em5 方程组
        func = self.get_func(stock.index, stock['em5'], 100)
        slope = self.get_slope(stock, func)
        if slope:
            result = self.conclusion(stock,slope[2], func)
            if result:
                days = stock['date'][list(chain(*result))]
                days = days[days>=(datetime.today() - timedelta(self.bf_day))]
                if not days.empty:
                    return symbol,str(days.dt.strftime('%Y-%m-%d').tolist())

    def save(self,result,name_dict):

        path = join(self.base_dir,'results',f'func_model.csv')
        save_data = pd.DataFrame(result,columns=['symbol','date_list'])
        save_data['stock_name'] = save_data.symbol.apply(lambda x:name_dict[x])
        save_data.to_csv(path,index=False)
        logger.info('拟合函数保存csv')

        write_db(save_data, table_name='stock_func',chunksize=1000,is_delete=False)
        logger.info('拟合函数入库成功')

    def run(self):
        st = time.time()
        logger.info('拟合函数模型开始计算')
        jobs,result = [],[]
        pool = Pool(6)
        df = self.load_data()
        _tqdm = tqdm(range(len(set(df.symbol))),ncols=90,desc='stock')
        name_dict = {k[1]:k[2] for k  in  df.drop_duplicates(['symbol','name'])[['symbol','name']].itertuples()}
        for symbol, stock in df.groupby('symbol'):
            jobs.append((symbol,pool.apply_async(self.execute,args=(symbol,stock))))
        pool.close()

        for symbol,job in jobs:
            try:
                data = job.get()
            except Exception as e:
                logger.error(f"{symbol}计算失败:{e}")
            else:
                if data:
                    result.append(data)
            _tqdm.update(1)

        self.save(result,name_dict)
        logger.info(f'拟合函数模型计算完成,用时:{time.time() - st}')

if __name__ == '__main__':
    fit(bf_day=8).run()