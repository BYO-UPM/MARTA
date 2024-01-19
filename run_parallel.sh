# Adjusted loop for distributing folds 0 to 9 across 4 GPUs
for GPU_ID in 0 1 2 3; do
    # Calculate start and end fold for each GPU
    START_FOLD=$((GPU_ID * 2)) # Start from fold 0
    END_FOLD=$((START_FOLD + 1))

    # Adjust for the last GPU to go up to fold 9
    if [ "$GPU_ID" -eq 3 ]; then
        END_FOLD=9
    fi

    for FOLD in `seq $START_FOLD $END_FOLD`; do
        ./run_fold.sh $FOLD $GPU_ID &
    done
done
wait
