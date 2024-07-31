#!/bin/bash

# Requirement: 8x H100 GPUs.


# Model: neuralmagic/Meta-Llama-3-70B-Instruct-FP8-KV 
# Query: 2048 input tokens, 11 output tokens, QPS 4, 500 requests
# Resource: 8x H100
# Approaches:
# 1. Chunked prefill: 1 vllm instance with tp=8
# 2. Chunked prefill: 2 vllm instance with tp=4, equivalent to 1 tp=4 instance with QPS 4
# 3. Disaggregated prefill: 1 prefilling instance and 1 decoding instance
# Prefilling instance: max_output_token=1
# Decoding instance: force the input tokens be the same across requests to bypass prefilling

set -ex

kill_gpu_processes() {
  # kill all processes on GPU.
  pkill pt_main_thread
  sleep 10

  # remove vllm config file
  rm -rf ~/.config/vllm

  # Print the GPU memory usage
  # so that we know if all GPU processes are killed.
  gpu_memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  # The memory usage should be 0 MB.
  echo "GPU 0 Memory Usage: $gpu_memory_usage MB"
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


benchmark() {

  # compare chunked prefill with disaggregated prefill

  results_folder="./results"
  model="neuralmagic/Meta-Llama-3-70B-Instruct-FP8"
  dataset_name="sonnet"
  dataset_path="../sonnet_4x.txt"
  num_prompts=500
  qps=$1
  prefix_len=64
  input_len=2048
  output_len=$2


  # chunked prefill with tp=4
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8000 \
    -tp 4 \
    --disable-log-stats \
    --disable-log-requests \
    --enable-chunked-prefill &
  wait_for_server 8000

  python3 ../benchmark_serving.py \
          --backend vllm \
          --model $model \
          --dataset-name $dataset_name \
          --dataset-path $dataset_path \
          --sonnet-input-len $input_len \
          --sonnet-output-len $output_len \
          --sonnet-prefix-len $prefix_len \
          --num-prompts $((num_prompts / 2)) \
          --port 8000 \
          --save-result \
          --result-dir $results_folder \
          --result-filename chunked_prefill_tp4.json \
          --request-rate $((qps / 2))
  kill_gpu_processes


  # disaggregated prefill
  # prefill with tp=4
  python3 -m vllm.entrypoints.openai.api_server \
          --model $model \
          --port 8000 \
          -tp 4 \
          --disable-log-stats \
          --disable-log-requests &
  wait_for_server 8000
  # set output-len to 1 so that it only do prefilling
  python3 ../benchmark_serving.py \
          --backend vllm \
          --model $model \
          --dataset-name $dataset_name \
          --dataset-path $dataset_path \
          --sonnet-input-len $input_len \
          --sonnet-output-len 1 \
          --sonnet-prefix-len $prefix_len \
          --num-prompts $num_prompts \
          --port 8000 \
          --save-result \
          --result-dir $results_folder \
          --result-filename disagg_prefill_tp4.json \
          --request-rate $qps
  kill_gpu_processes

  # decode with tp=4, enable APC
  python3 -m vllm.entrypoints.openai.api_server \
          --model $model \
          --port 8000 \
          -tp 4 \
          --enable-prefix-caching \
          --disable-log-stats \
          --disable-log-requests &
  wait_for_server 8000
  # skip prefilling 
  # by enabling APC and force the input tokens be the same
  python3 ../benchmark_serving.py \
          --backend vllm \
          --model $model \
          --dataset-name $dataset_name \
          --dataset-path $dataset_path \
          --sonnet-input-len $input_len \
          --sonnet-output-len $output_len \
          --sonnet-prefix-len $input_len  \
          --num-prompts $num_prompts \
          --port 8000 \
          --save-result \
          --result-dir $results_folder \
          --result-filename disagg_decode_tp4.json \
          --request-rate $qps
  kill_gpu_processes

  python3 analyze_benchmark_results.py \
          --results-folder $results_folder \
          --output-len $output_len \
          --qps $qps

}


main() {

  (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  (which jq) || (apt-get -y install jq)
  (which socat) || (apt-get -y install socat)

  cd "$(dirname "$0")"

  cd ..
  # create sonnet-4x.txt
  echo "" > sonnet_4x.txt
  for _ in {1..4}
  do
    cat sonnet.txt >> sonnet_4x.txt
  done
  cd disagg_benchmarks

  rm -rf results
  mkdir results

  default_qps=4
  default_output_len=150

  for target_qps in 2 4 8 16
  do
    benchmark $target_qps $default_output_len
  done

  # for target_output_len in 5 10 20 40 80
  # do
  #   benchmark $default_qps $target_output_len
  # done

}


main "$@"
