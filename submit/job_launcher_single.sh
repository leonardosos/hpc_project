#!/bin/bash

echo " -------- START ---------"

# Loop through numbers 1 to 16
for num in {1..5}; do
    echo
    echo "Submit script_job_$num.sh"

    qsub script_job_"$num".sh
        
done

# Wait 45 seconds 
sleep 45

for num in {6..12}; do
    echo
    echo "Submit script_job_$num.sh"

    qsub script_job_"$num".sh
        
done

# Wait 45 seconds 
sleep 45

for num in {12..16}; do
    echo
    echo "Submit script_job_$num.sh"

    qsub script_job_"$num".sh
        
done


echo " -------- DONE ---------"