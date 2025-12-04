#!/usr/bin/env bash

# Run MARTA_Supervised_international.py across folds and domain settings, always
# keeping GPUs 0 and 1 busy. As soon as a job on a GPU finishes, the next pending
# configuration is dispatched to that GPU until all runs complete.

set -u

SCRIPT="MARTA_Supervised_international.py"
LATENT_DIM=3
FOLDS=($(seq 0 10))
DOMAIN_SETTINGS=(0 1)
GPUS=(0 1)
LOG_DIR="logs/marta_supervised_international"

mkdir -p "$LOG_DIR"

# Build queue of jobs (fold/domain combinations)
declare -a JOB_QUEUE=()
for fold in "${FOLDS[@]}"; do
    for domain in "${DOMAIN_SETTINGS[@]}"; do
        JOB_QUEUE+=("${fold}:${domain}")
    done
done

total_jobs=${#JOB_QUEUE[@]}
next_job=0
completed_jobs=0

declare -A GPU_PIDS
declare -A GPU_DESC

launch_job() {
    local gpu=$1
    local job_spec=$2
    IFS=":" read -r fold domain <<<"$job_spec"

    local log_file="${LOG_DIR}/fold_${fold}_domain_${domain}_gpu_${gpu}.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching fold ${fold}, domain ${domain} on GPU ${gpu}. Log: ${log_file}"

    python "$SCRIPT" \
        --fold "$fold" \
        --gpu "$gpu" \
        --latent_dim "$LATENT_DIM" \
        --domain_adversarial "$domain" \
        >"$log_file" 2>&1 &

    local pid=$!
    GPU_PIDS[$gpu]=$pid
    GPU_DESC[$gpu]="fold ${fold} domain ${domain}"
}

check_finished_jobs() {
    for gpu in "${GPUS[@]}"; do
        local pid="${GPU_PIDS[$gpu]:-}"
        [[ -z "$pid" ]] && continue

        if ! kill -0 "$pid" 2>/dev/null; then
            # Process ended; reap it and report status
            if wait "$pid"; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed ${GPU_DESC[$gpu]} on GPU ${gpu}."
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: ${GPU_DESC[$gpu]} on GPU ${gpu} failed." >&2
            fi
            unset GPU_PIDS[$gpu]
            unset GPU_DESC[$gpu]
            ((completed_jobs++))
        fi
    done
}

while (( completed_jobs < total_jobs )); do
    check_finished_jobs

    for gpu in "${GPUS[@]}"; do
        if [[ -z "${GPU_PIDS[$gpu]:-}" ]] && (( next_job < total_jobs )); then
            launch_job "$gpu" "${JOB_QUEUE[$next_job]}"
            ((next_job++))
        fi
    done

    # If there are still running jobs, wait a moment before polling again
    if (( completed_jobs < total_jobs )); then
        sleep 5
    fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All ${total_jobs} jobs completed."
