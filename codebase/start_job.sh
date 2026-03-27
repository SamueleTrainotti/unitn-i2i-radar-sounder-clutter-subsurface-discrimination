# Check for GPU availability
./check_gpu.sh
if [ $? -ne 0 ]; then
	echo "GPU check failed. Exiting."
	exit 1
fi
echo "Starting hyperparameter tuning...\n"
docker exec samuele_samueletrainotti python scripts/tune_hyperparameters.py -c scripts/config_files/config.yaml -t scripts/config_files/tune_config.yaml 

# TODO add gitignore rules
# TODO check error in loading not existing cached data/checkpoint
# consider moving data in another folder