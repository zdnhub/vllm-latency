#!/bin/bash

# This script should be run inside the vllm container. Enter the latest vllm container by
# docker run -it --runtime nvidia --gpus all --env "HF_TOKEN=<your HF TOKEN>"  --entrypoint /bin/bash  vllm/vllm-openai:latest
# (please modify `<your HF TOKEN>` to your own huggingface token in the above command
# Then, run the following command:
# 
# Then, copy-paste this file into the docker (any path in the docker works) and execute it using bash.
# Benchmarking results will be at /vllm/benchmarks/results/benchmark_results.md


set -e
set -o pipefail


check_gpus() {
  declare -g gpu_count=$(nvidia-smi --list-gpus | wc -l)

  if [[ $gpu_count -gt 0 ]]; then
    echo "GPU found."
  else
    echo "Need at least 1 GPU to run benchmarking."
    exit 1
  fi

  declare -g gpu_type=$(echo $(nvidia-smi --query-gpu=name --format=csv,noheader) | awk '{print $2}')
  echo "GPU type is $gpu_type"
}


check_hf_token() {
  if [[ -z "$HF_TOKEN" ]]; then
    echo "Error: HF_TOKEN is not set."
    exit 1
  elif [[ ! "$HF_TOKEN" =~ ^hf_ ]]; then
    echo "Error: HF_TOKEN does not start with 'hf_'."
    exit 1
  else
    echo "HF_TOKEN is set and valid."
  fi
}

json2args() {
  # transforms the JSON string to command line args, and '_' is replaced to '-'
  # example:
  # input: { "model": "meta-llama/Llama-2-7b-chat-hf", "tensor_parallel_size": 1 }
  # output: --model meta-llama/Llama-2-7b-chat-hf --tensor-parallel-size 1
  local json_string=$1
  local args=$(
    echo "$json_string" | jq -r '
      to_entries |
      map("--" + (.key | gsub("_"; "-")) + " " + (.value | tostring)) |
      join(" ")
    '
  )
  echo "$args"
}

wait_for_server() {
  # wait for vllm server to terminate
  timeout 600 bash -c 'until curl localhost:8000/v1/completions; do sleep 1; done' || exit 1
}

kill_gpu_processes() {
  # kill vllm instances
  # List all the processes using the GPU and get their PIDs
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)

  # Check if there are any PIDs to kill
  if [ -z "$pids" ]; then
      echo "No GPU processes found."
  else
      # Kill each process
      for pid in $pids; do
          kill -9 $pid
          echo "Killed process with PID: $pid"
      done

      echo "All GPU processes have been killed."
  fi

  # waiting for GPU processes to be fully killed
  sleep 10

  # Print the GPU memory usage
  # so that we know if all GPU processes are killed.
  gpu_memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  echo "GPU 0 Memory Usage: $gpu_memory_usage MB"

}


check_gpus
check_hf_token


(which wget && which curl) || (apt-get update && apt-get install -y wget curl)
# jq is the JSON parser to parse the parameter files
(which jq) || (apt-get update && apt-get -y install jq)

# get the current IP address, required by benchmark_serving.py
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

# prepare for benchmarking
cd benchmarks
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
RESULTS_FOLDER=results/
SERVING_TESTS=../.buildkite/nightly-benchmarks/serving-tests.json
LATENCY_TESTS=../.buildkite/nightly-benchmarks/latency-tests.json
POSTPROCESS_SCRIPT=../.buildkite/nightly-benchmarks/results2md.py
mkdir -p $RESULTS_FOLDER



# Iterate over latency tests
jq -c '.[]' $LATENCY_TESTS | while read -r params; do

  # get the test name, and append the GPU type back to it.
  test_name=$(echo $params | jq -r '.test_name')_${gpu_type}
  if [[ ! "$test_name" =~ ^latency_ ]]; then
    echo "In latency-test.json, test_name must start with \"latency_\"."
    exit 1
  fi

  # get client and server arguments
  latency_params=$(echo $params | jq -r '.parameters')
  latency_args=$(json2args "$latency_params")

  # check if there is enough GPU to run the test
  tp=$(echo $latency_params | jq -r '.tensor_parallel_size')
  if [[ $gpu_count -lt $tp ]]; then
    echo "Required tensor-parallel-size $tp but only $gpu_count GPU found. Skip testcase $testname."
    continue
  fi

  latency_command="python3 benchmark_latency.py \
    --output-json $RESULTS_FOLDER/${test_name}.json $latency_args"

  echo "Running test case $test_name"
  echo "Latency command: $latency_command"
  # record the benchmarking commands
  echo $(
    jq -n \
      --arg latency "$latency_command" \
      '{
        latency_command: $latency,
      }'
  ) > $RESULTS_FOLDER/$test_name.commands

  # run the benchmark
  eval $latency_command

  kill_gpu_processes

done






# Iterate over serving tests
jq -c '.[]' $SERVING_TESTS | while read -r params; do


  # get the test name, and append the GPU type back to it.
  test_name=$(echo $params | jq -r '.test_name')_${gpu_type}
  if [[ ! "$test_name" =~ ^serving_ ]]; then
    echo "In serving-test.json, test_name must start with \"serving_\"."
    exit 1
  fi

  # get client and server arguments
  server_params=$(echo $params | jq -r '.server_parameters')
  client_params=$(echo $params | jq -r '.client_parameters')
  server_args=$(json2args "$server_params")
  client_args=$(json2args "$client_params")


  # check if there is enough GPU to run the test
  tp=$(echo $server_params | jq -r '.tensor_parallel_size')
  if [[ $gpu_count -lt $tp ]]; then
    echo "Required tensor-parallel-size $tp but only $gpu_count GPU found. Skip testcase $testname."
    continue
  fi

  server_command="python3 \
    -m vllm.entrypoints.openai.api_server \
    $server_args"

  client_command="python3 benchmark_serving.py \
    --backend vllm \
    --save-result \
    --result-dir $RESULTS_FOLDER \
    --result-filename ${test_name}.json \
    $client_args"

  echo "Running test case $test_name"
  echo "Server command: $server_command"
  echo "Client command: $client_command"
  # record the benchmarking commands
  echo $(
    jq -n \
      --arg server "$server_command" \
      --arg client "$client_command" \
      '{
        server_command: $server,
        client_command: $client
      }'
  ) > $RESULTS_FOLDER/$test_name.commands

  # run the server
  eval $server_command &
  server_pid=$!

  # wait until the server is alive
  wait_for_server

  # run the client
  eval $client_command

  # clean up
  kill_gpu_processes

  # debugging memory leaking
  used_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)

  echo "GPU0 used memory: $used_memory MB"


done

# postprocess benchmarking results
pip install tabulate pandas
python3 ../.buildkite/nightly-benchmarks/results2md.py

# if the agent binary is not found, skip uploading the results, exit 0
if [ ! -f /workspace/buildkite-agent ]; then
    exit 0
fi

# upload the results to buildkite
/workspace/buildkite-agent annotate --style "info" --context "benchmark-results" < $RESULTS_FOLDER/benchmark_results.md

# upload artifacts
/workspace/buildkite-agent artifact upload "$RESULTS_FOLDER/*"
