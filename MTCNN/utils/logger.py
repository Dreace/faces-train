import logging

logger = logging.getLogger("root")
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter(fmt='[%(asctime)s %(filename)s:%(lineno)d %(funcName)s] [%(levelname)s] %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
