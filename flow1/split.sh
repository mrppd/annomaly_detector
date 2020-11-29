#!/bin/sh

j=1
len=$4
for i in $(seq $2 $len $3)
do
   echo $j
   sed -n "$i,+$(($len-1))p" $1 > $j
   #head -n 10 $1 
   j=$((j + 1))  
done
