# AI Audio Engineer
## Training an AI to learn Audio Mixing and Music Production

### Setup
- `$ chmod +x packages.sh`
- `$ ./packages.sh`
- `$ pip install -r requirements.txt`

### Training
- Place the dataset in a directory on the project root named 'dataset'
- Go to architecture and run 
  ```
  python model_train.py --datapath ../dataset
  ```
