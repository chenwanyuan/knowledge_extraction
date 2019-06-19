#!/bin/sh
#根据进程名杀死进程


PROCESS=`ps -ef|grep 'python' | grep 'server.py' | grep '7070' | grep -v grep|grep -v PPID|awk '{ print $2}'`

if [[ $PROCESS == '' ]]
then
    echo '没有 python server.py 这个进程'
fi

for i in $PROCESS
do
    echo "Kill the python server.py process [ $i ]"
    kill -9 $i
    if [ $? -ne 0 ]
    then
        echo "杀死进程 [ $i ] ,失败"
    else
        echo "杀死进程 [ $i ] ,成功"
    fi
done