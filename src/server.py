#!/usr/bin/python3
#--coding:utf-8--
#@file : server.py
#@author: chenwanyuan
#@date:2019/06/19/
#@description:
#@other:

from optparse import OptionParser
import os

from mysetting import *

import sys
import tornado
import tornado.web
import tornado.ioloop
import tornado.httpserver
from tornado.process import task_id

import json
import datetime


import traceback
from process import TextProcess
from common.log.log import log_base_config
import logging



#g_textprocess = TextProcess(MODEL_PARAM)


#class InputData():
#    def __init__(self,sequence_id,partnercode,event,content):
#        self.sequence_id = sequence_id
#        self.partnercode = partnercode
#        self.event = event
#        self.content = content
class JRes():
    '''
    JRes json的返回数据格式
    '''
    @staticmethod
    def error(code=-1,msg="error"):
        return JRes.set(code,msg)
    @staticmethod
    def set(code,msg):
        return JRes._init_reponse(code,msg)
    @staticmethod
    def _init_reponse(code,msg):
        return {
            'status_code':code,
            "status_msg":msg
        }
    @staticmethod
    def add_data(data,time,sequence_id=""):
        res = JRes.set(0,"ok")
        res["datas"] = data
        res["time"] = time
        if sequence_id:
            res["sequence_id"] = sequence_id
        return res

class KEProcess(tornado.web.RequestHandler):
    '''
    的处理进程
    '''
    def initialize(self,text_process):
        self.text_process = text_process
        #print(self.textcnn)
    @tornado.web.asynchronous
    def get(self):
        self._do()

    @tornado.web.asynchronous
    def post(self):
        self._do()

    def _do(self):
        self.set_header("Content-Type","application/json")

        try:
            start_time = datetime.datetime.now()
            sequence_id = self.get_argument("sequence_id", "")
            content = self.get_argument("content", "")
            end_time = datetime.datetime.now()

            time = (end_time - start_time).microseconds / 1000
            logging.info('rt=%f,len=%d' % (time,len(content)))

            input_data = {"sequence_id":sequence_id}
            input_data['content'] = content

            logging.info(input_data)
            start_time = datetime.datetime.now()
            result = self._do_imp(input_data)
            end_time = datetime.datetime.now()
            time = (end_time - start_time).microseconds / 1000
            result_data = JRes.add_data(result,time)
            logging.info(result_data)
            self.write(result_data)
            self.finish()
        except Exception as e:
            #print(traceback.format_exc())
            result_data = JRes.error(msg=str(traceback.format_exc()))
            logging.error(result_data)
            self.write(result_data)
            self.finish()

    def _do_imp(self,input_data):
        #print(input_data.__dict__)
        result = self.text_process.process(input_data['content'])
        return result




class Server():
    '''
    tornado.web 服务
    '''
    def __init__(self,port,process_num = 1):
        self.port = port
        self.process_num = process_num
        self.text_process = TextProcess(MODEL_PARAM)
        self.application = tornado.web.Application([
        (r"/ke", KEProcess,dict(text_process=self.text_process))
        ],debug=False,static_path=os.path.join(os.path.dirname(__file__),u"static"))
    def run(self):
        self.server = tornado.httpserver.HTTPServer(self.application)
        self.server.bind(port=self.port)
        self.server.start(self.process_num)

        id = task_id()
        #多进程并不是多线程，需要考虑日志半夜切分问题，所以根据进程记录区分日志。
        if id != None:
            log_base_config(LOG_NAME,"{}/{}/{}".format(LOG_DIR,id,LOG_FILE),level=logging.INFO)
            #延后加载，否则tensoflow 产生冲突
            self.text_process.init()
            self.text_process.process('')
        else:
            log_base_config(LOG_NAME,"{}/{}".format(LOG_DIR,LOG_FILE))
            self.text_process.init()
            self.text_process.process('')

        tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    import jieba
    from optparse import OptionParser

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = OptionParser()
    parser.add_option("--port",type="int", dest="port",default = SERVER_PORT,help="端口")
    parser.add_option("--process", type="int", dest="process_num", default=SERVER_PROCESS_NUM,help="进程数")
    (options, args) = parser.parse_args()
    print(" ".join(jieba.cut("我草你妈")))
    server = Server(options.port,options.process_num)
    server.run()
