import asyncio
import functools
import time
from contextlib import contextmanager
from datetime import datetime
from glob import glob

import pymysql
import pandas as pd

from .logger import logger
from os.path import abspath,dirname,join

BASE_DIR = dirname(dirname(abspath(__file__)))



def load_df():
    base_dir = dirname(dirname(abspath(__file__)))
    file_list = glob(join(base_dir, 'history_data', '*.pkl'))
    print(file_list[-1:])
    file_path = sorted(file_list)[-1]
    df = pd.read_pickle(file_path)
    df["date"] = df.date.astype("datetime64")
    df.sort_values(["symbol", "date"], inplace=True, ignore_index=True)
    df["turnover"] = df.volume * df.close * 100
    return df


def collection_error(single=True):
    def action(func):
        def hander(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(e)
                if single:
                    raise e
        return hander
    return action

def get_sql(columns, table_name):
    # 获取sql
    #     columns = get_sys_guid_col(table_name, columns, con)
    insert_col = ','.join(columns)
    sql_col = '%' + ',%'.join('s' * len(columns))
    sql = f"insert into {table_name}({insert_col}) values({sql_col})"
    return sql

def timeit(function):
    if asyncio.iscoroutinefunction(function):
        @functools.wraps(function)
        async def wrapped_async(*args, **kwargs):
            stime = time.time()
            result = await function(*args, **kwargs)
            etime = time.time()
            logger.info(f"{function.__name__} 运行了 {etime-stime}s")
            return result

        return wrapped_async

    else:

        @functools.wraps(function)
        def wrapped_sync(*args, **kwargs):
            stime = time.time()
            result = function(*args, **kwargs)
            etime = time.time()
            logger.info(f"{function.__name__} 运行了 {etime-stime}s")
            return result

        return wrapped_sync

@contextmanager
def timeit_manager(msg, ):
    try:
        st = time.time()
        yield
    except Exception as e:
        raise e
    else:
        logger.info("{} use time {}".format(msg, time.time() - st))

@collection_error(single=False)
def write_db(df, table_name='',con=None,chunksize=30000,is_delete=True):
    if con is None:
        con = pymysql.connect(user='root', port=3306, host='www.chaoyue.red', password='123456', db='bbs_1')

    db = con.cursor()

    where =  f"where sw_class='{df.sw_class.iloc[0]}'" if is_delete else ''
    db.execute(f"delete from {table_name} {where}")
    con.commit()

    df['create_date'] = datetime.now()

    c = [(column, str(date)) for column, date in zip(df.columns.tolist(), df.dtypes) if 'date' in str(date)]
    for column, date in c:
        df[column] = df[column].astype('str')
        df.replace('NaT', None, inplace=True)

    df = df.mask(df.isna(),None)
    sql = get_sql(df.columns,table_name)
    avg_list = [(i, i + chunksize) for i in range(0, len(df) + chunksize, chunksize)]
    for i,k in avg_list:
        data = df.iloc[i:k].values.tolist()
        try:
            if data:
                db.executemany(sql,data)
        except Exception as e:
            logger.exception(e)
            con.rollback()
            raise e
        else:
            con.commit()