#!/bin/bash



while true; do

./Generator/Linux/LogoDatabaseRenderer.x86_64 +objects 1000 +h 512 +w 512 +images 64 &

sleep 3600

pkill -f LogoDatabaseRend
echo 'reset'

done
