import warnings
from datetime import timedelta
import numpy as np
import pandas as pd


class Average(object):

    def __init__(self, line_count, var, col, bf_day=6):
        self.var = var
        self.line_count = line_count
        self.bf_day = bf_day
        self.col = col
        self.info_col = ['jx_date', 'jx_names', 'jx_counts', 'var']
        self.result = []

    def get_junxian_col(self, data):
        for i in self.col:
            if i not in ['ma5', 'ma10', 'ma20']:
                data[i] = data['close'].rolling(int(i.replace('ma', '')), min_periods=1).mean()
        return data

    def get_slope(self, df):
        y = df.tolist()
        x = list(range(1, len(y) + 1))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                xielv =  np.polyfit(x, y, 1)[0]
            except Exception as e:
                pass
        return xielv

    def jx_var(self, df):
        mean = df.mean()
        return (df - mean).abs().mean() / mean

    def execute(self, symbol, data):
        data = self.get_junxian_col(data)
        jx_df = data[self.col]
        df = self.get_ma_A(jx_df,symbol)
        origin_df = df.merge(data, left_on=['jx_date', 'symbol'], right_on=['t_date', 'symbol'],suffixes=('','_y'))
        origin_df.drop([i+'_y' for i in self.col],axis=1,inplace=True)
        origin_df['bf_close'] = ((origin_df['close'] / origin_df['close'].shift(self.bf_day)) - 1).round(2)
        origin_df['af_close'] = (origin_df['close'].shift(-self.bf_day) / origin_df['close'] - 1).round(2)

        date_range = jx_df.apply(func=self.jx_var, axis=1)
        origin_df['var'] = date_range.tolist()

        date_range = date_range[date_range < self.var].sort_index()
        df_val = pd.DataFrame(date_range,columns=['var']).reset_index().rename(columns={'date':'jx_date'})
        result = df_val.merge(origin_df,on=['jx_date'],suffixes=('','_y'))
        result.drop(['var_y'],axis=1,inplace=True)
        result = result.mask(result.isna(),None)
        # for date, var in date_range.iteritems():
        #     print(self.result[-1])
        #     last_date = self.result[-1].jx_date[-1] if self.result else None
        #     print(last_date,date)
        #     if last_date and (date - last_date).days <= 7:
        #         continue
        #     self.result.append(self.model(jx_df, date, var))
        return result,origin_df

    def get_ma_A(self, df,symbol):
        '''
        20日线 和 10日线 在 当日前 N 个交易日内 有上升趋势
        且 30 日线 呈上升趋势
        :param df: stock df
        :param st: 交易日开始
        :param en: 当日
        :return: 是否满足均线条件
        '''
        # st = (date + timedelta(days=-self.bf_day)).date().strftime('%Y-%m-%d')
        slope = df.rolling(self.bf_day, min_periods=1).apply(self.get_slope)
        slope['jx_counts'] = (slope > 0).sum(axis=1)
        slope = slope.round(2)

        slope['jx_names'] = slope['ma5'].astype('str') + ',' \
                            + slope['ma10'].astype('str') + ',' \
                            + slope['ma20'].astype('str') + ',' \
                            + slope['ma30'].astype('str')

        slope['symbol'] = symbol
        slope.reset_index(inplace=True,drop=False)
        slope.rename(columns={'date':'jx_date'},inplace=True)
        return slope


class EMAverage(Average):

    def get_junxian_col(self, data):
        for i in self.col:
            data[i] = data['close'].ewm(span=int(i.replace('ma', '')), ignore_na=False, adjust=True).mean()
        return data