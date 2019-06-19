#!/usr/bin/python3
#--coding:utf-8--
#@file       : client.py
#@author     : chenwanyuan
#@date       :2019/06/19/
#@description:
#@other      :

import requests

content = '雏鹰农牧(002477)复牌大跌8% 澄清造假质疑投资者不买账通产丽星(002243)股东中'
url = 'http://localhost:7070/ke?'
print(requests.post(url=url,data={'content':content}))

