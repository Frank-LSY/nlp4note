#!/bin/sh
pids=`/sbin/pidof $1`
cpunum=`cat /proc/cpuinfo | grep processor | wc -l`
 
cpuidx=0
for pid in $pids
do
    /usr/bin/taskset -cp ${cpuidx} ${pid}
    cpuidx=$(($cpuidx+1))
    cpuidx=$(($cpuidx%$cpunum))
    echo $cpuidx
done