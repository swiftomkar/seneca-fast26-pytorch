#!/bin/bash

# Define an array of Redis ports
declare -a redis_ports=("6376" "6378" "6380" "6388" "6389" "6390") # Add your Redis ports here

# Iterate through each port and flush all keys
for port in "${redis_ports[@]}"
do
    echo "Flushing Redis on port $port"
    redis-cli -p "$port" FLUSHALL
done

echo "Flush operation completed for all Redis databases on specified ports"
