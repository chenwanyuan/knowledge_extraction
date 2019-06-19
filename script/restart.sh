#!/bin/sh
DIR=$PWD

basepath=$(cd `dirname $0`; pwd)
cd ${basepath}

bash stop.sh
bash start.sh

cd $DIR
