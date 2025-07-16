#!/bin/bash

# Check if enough arguments are provided
if [ "$#" -ne 6 ]; then
    echo "Please provide maxmemory values for three different ports."
    exit 1
fi

# Array of Redis ports (Change these to match your Redis ports)
declare -a redis_ports=("$1" "$3" "$5")

# Retrieve maxmemory values for each port's database
maxmemory_values=("$2" "$4" "$6")

# Iterate through each Redis port
for ((i=0; i<${#redis_ports[@]}; i++))
do
    port="${redis_ports[$i]}"
    maxmemory_value="${maxmemory_values[$i]}"

    echo "Setting maxmemory $maxmemory_value for database on port $port"
    redis-cli -p "$port" CONFIG SET maxmemory "$maxmemory_value"
done

echo "Maxmemory configuration updated for all databases on specified ports"
