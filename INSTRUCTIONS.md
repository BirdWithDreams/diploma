```bash
# Clone repository
git clone https://github.com/BirdWithDreams/diploma
cd diploma

# Create virtual env in any way you like. Python must be 3.11
# Install requirements
pip install -r requirements.txt

# Create dir for checkpoints
mkdir checkpoints
cd checkpoints

# Copy `finale_models.tar.gz` in `checkpoints` and unzip it.
cp <path_to_finale_models.tar.gz> .
tar -xvf finale_models.tar.gz

# Then mode to data directory
cd ../data
# Copy `data.tar.gz` to `data` and unzip it.
cp <path_to_data.tar.gz> .
tar -xvf data.tar.gz

# Move to the script directory
cd ../my_scripts/dpo_training

# Add project directory to the `PYTHONPATH` and run script.
export PYTHONPATH=$PYTHONPATH:<path_to_project_directory>
python dpo_entry_point.py --num-threads 16 # You may increase `--num-threads` parameter if machine allows it. `--num-threads` parameter set the number of parallel model runs.
```

After script finished
```bash
cd ../../data/dpo_dataset
tar -cvzf dpo_data.tar.gz ./vctk-asr ./vctk-asr-gen ./lg-asr ./lg-asr-gen
```

`dpo_data.tar.gz` is the finale archive.

