
#!/bin/bash

# Function to kill script
cleanup() {
    echo "Exiting script..."
    exit 0
}

trap cleanup SIGINT

# arguments
num_layouts=$1
num_runs=$2
num_training=$3

# Parameters
mean_dim=8
std_dev_dim=4

layout_widths=()
layout_heights=()


# agents
agents=("ApproximateQAgent" "ReinforceAgent" "ActorCriticAgent")


# clearing existing layout and results
rm -r layouts
rm -r results
# setting up layouts and results dir
mkdir layouts
mkdir results
for agent in "${agents[@]}"; do
    mkdir results/$agent
done



# Function to generate layout using layoutGenerator.py
generate_layout() {
    python layoutGenerator.py -g $1 -m $2 -c $3 -f $4
}

# Function to test RL agent on a layout
train_agent() {
    python pacman.py -p $1 -l $2 -q -n $3 -x $4 -k $5 -a filename=${6}_k$5
}

layouts=()


# Generating 100 layouts
for ((i=0; i<num_layouts; i++)); do
    width=$(python -c "import numpy as np; print(max(abs(int(np.random.normal($mean_dim, $std_dev_dim))*2),6))")
    
    max_height=$((width/2))
    height=$(python -c "import numpy as np; print(max(int(np.random.randint(0, $max_height)),3))")

    capsules=$(python -c "import numpy as np; print(int(np.random.randint(1, 6)))")

    timestamp=$(date +"%Y%m%d%S")
    layout_name="layout_${timestamp}_${i}"
    layouts+=($layout_name)

    generate_layout $width $height $capsules ${layout_name}.lay
    echo "Layout generated -g $width -m $height -c $capsules"
done


# Testing all RL agents on each layout in parallel
for layout in "${layouts[@]}"; do
    num_ghost=$(python -c "import numpy as np; print(int(np.random.randint(2, 4)))")
    for agent in "${agents[@]}"; do
        echo ""
        echo "Training ${agent} on layout ${layout}"
        train_agent $agent $layout $num_runs $num_training $num_ghost $layout &
    done
    wait # Wait for all agents to finish training for the current layout
done

# Wait for all agents to finish training on all layouts
wait
