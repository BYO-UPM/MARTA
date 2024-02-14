# Adjusted loop for distributing folds 0 to 9 across GPUs 0 and 1
for FOLD in `seq 0 10`; do
    # Assign GPU 0 for even folds and GPU 1 for odd folds
    GPU_ID=$((FOLD % 2))
    MASKED=7

    ./run_fold.sh $FOLD $GPU_ID $MASKED &

    # Wait after launching two processes on the same GPU
    if [ $((FOLD % 4)) -eq 1 ] || [ $((FOLD % 4)) -eq 2 ]; then
        wait
    fi
done

# Wait for the last set of processes to finish
wait
