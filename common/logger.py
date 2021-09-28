import logging
from os.path import dirname,join,abspath
logger = logging.getLogger(__name__)
path = join(dirname(dirname(abspath(__file__))),'log','info.log')
formart = logging.Formatter(
    "[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s"
)
hander_std = logging.StreamHandler()
hander_std.setFormatter(formart)
hander = logging.FileHandler(path,encoding='utf-8')
hander.setLevel(logging.INFO)
hander.setFormatter(formart)
logger.addHandler(hander)
logger.addHandler(hander_std)
logger.setLevel(logging.INFO)
