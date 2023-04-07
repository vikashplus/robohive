# Generate dataset for each policy inside the specified dir

# gather inputs
env_name=$1    # Env name to rollout the policies
policy=$2     # directories to search for policies
n_trails=$3    # number of trials to rollout for each policy

# Search for policies and rollout
# echo === Finding policies in dir: $pol_dir
# for policy in $(find $pol_dir -name '*.pickle');
# do
    # find out the seed for the found policy
# if [[ $policy =~ seed=([0-9]+) ]]; then
#     seed="${BASH_REMATCH[1]}"
# else
#     echo "Couldn't determine seed from the $policy"
# fi

# configure output path
output_dir=./../dataset/$env_name/
mkdir -p $output_dir

# announce the details before rollouts and copy the policy
echo Roling out $n_trails Trials using policy:$policy with seed:$seed on env:$env_name
echo Output: $output_dir
cp $policy $output_dir

# rollout the trajectories
sim_backend=MUJOCO python examine_env.py \
    --env_name $env_name \
    --policy_path $policy \
    --num_episodes $n_trails \
    --render none \
    --save_paths True \
    --render_visuals True \
    --output_dir $output_dir \
    --output_name $env_name
# done
