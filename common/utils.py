from datetime import datetime

import pymysql

from .logger import logger
from os.path import abspath,dirname,join

BASE_DIR = dirname(dirname(abspath(__file__)))


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