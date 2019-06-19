import common.log.log as log
import logging
import time
log.log_base_config("root","./log.log")
print("AA")
def test():
    #log.basicConfig(filename="./log.log",level=log.INFO,format='[%(asctime)s][%(funcName)s][%(levelname)s][%(process)d][%(thread)d][%(message)s]')
    while True:
        logging.info("Fuck:{}".format(time.time()))
        time.sleep(0.1)

if __name__ == '__main__':
    test()