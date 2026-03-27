# Check for GPU availability
./check_gpu.sh
if [ $? -ne 0 ]; then
	echo "GPU check failed. Exiting."
	exit 1
fi
echo "Starting training...\n"
docker exec samuele_samueletrainotti python scripts/train.py -c scripts/config_files/config.yaml