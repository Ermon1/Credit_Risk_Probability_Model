from src.utility.config_loader import config 
from src.data_processing import process_and_save_data

# Load YAML
cfg = config.load('data.yaml')  

# Process raw data and save processed features
df_features = process_and_save_data(cfg)

# Inspect
df_features.head()   
