# POV
Project for POV - face verification from image

## Usage
### Preparing data CASIA-WebFace data
```bash
python prepare_data.py -d $DATA_DIR
```

### Training model
```bash
python train_model.py -d $INPUT_PREPARED_DATA_DIR -D $OUTPUT_DIR
```
