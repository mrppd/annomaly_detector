#!/bin/sh
sbatch -t 48:00:00 --wrap="bash split.sh ../auth.txt 1 175238409" 


