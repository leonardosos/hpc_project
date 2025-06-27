#!/bin/bash

echo " -------- START ---------"

# Loop through numbers 1 to 16
for num in {1..16}; do
    echo
    echo "Submit script_job_$num.sh"

    qsub script_job_"$num".sh
    
    # Wait 45 seconds 
    sleep 45
        
done

echo " -------- DONE ---------"