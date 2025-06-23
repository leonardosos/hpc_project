#!/bin/bash

while true; do
    # Loop through numbers 1 to 16
    for num in {1..16}; do
        echo "Processing script_job_$num.sh"
        
        # Execute qsub 8 times in a row, twice for each number
        for round in {1..2}; do
            echo "Round $round for script_job_$num.sh"
            for i in {1..8}; do
                qsub script_job_"$num".sh
                echo "Submitted script_job_$num.sh (round $round, submission $i)"
            done
        done
    done
    
    echo "Waiting 45 seconds before next cycle..."
    sleep 45
done