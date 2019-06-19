#!/bin/sh
DIR=$PWD

basepath=$(cd `dirname $0`; pwd)
cd ${basepath}/../src


count=`ps -ef | grep 'python' | grep 'server.py' | grep '7070' |grep -v "grep" |wc -l`

if [ 0 == $count ]
then
    ulimit -c unlimited
    nohup python server.py --port 7070 --process 1 > err.log 2>&1 &
    sleep 20

    count=`ps -ef | grep 'python' | grep 'server.py' | grep '7070' |grep -v "grep" |wc -l`
    if [ 0 == $count ]
    then
        echo "服务启动失败"
    else
        echo "服务启动成功"
    fi
else
    echo "服务已经存在,不需要再次启动"
fi


cd $PWD




