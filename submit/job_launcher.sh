#!/bin/bash

echo " -------- START ---------"

# Loop through numbers 1 to 16
for num in {1..16}; do
    echo
    echo
    echo "Processing script_job_$num.sh"
    
    # Execute qsub 8 times in a row, twice for each number
    for round in {1..3}; do
        echo 
        echo "Round $round for script_job_$num.sh"
        echo
        for i in {1..8}; do
            echo "Submit script_job_$num.sh (round $round, submission $i)"
            qsub script_job_"$num".sh
        done
    sleep 45
    done
done

echo " -------- DONE ---------"
