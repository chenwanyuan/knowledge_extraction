#!/usr/bin/python3
#--coding:utf-8--
#@file       : log.py
#@author     : chenwanyuan
#@date       :2019/06/10/
#@description:
#@other      :

import logging
import os
from logging.handlers import TimedRotatingFileHandler
import os
import json

def log_base_config(servername="root",logfile="./test.log",level=logging.INFO,stream=False):
    logging.root.setLevel(level)
    formatter = logging.Formatter("[%(asctime)s][{}][%(process)d][%(thread)d][%(levelname)s][%(filename)s][%(funcName)s][%(lineno)d][%(message)s]".format(servername))
    if stream:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logging.root.addHandler(ch)
    log_dir = os.path.dirname(logfile)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
        #os.system("mkdir -p {}".format(log_dir))
    fileTimeHandler = TimedRotatingFileHandler(logfile, "midnight", 1, 30)
    fileTimeHandler.setFormatter(formatter)
    logging.root.addHandler(fileTimeHandler)