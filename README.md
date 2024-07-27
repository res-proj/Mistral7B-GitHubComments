# Mistral7B  
This repository contains the code for fine-tuning Mistral-7B-Instruct pre-trained model.  
## How to use  
1. Install requirements: python -m pip install -r requirements.txt
2. Install PyTorch: https://pytorch.org/get-started/locally/ (based on your device, choose the corresponding version)
3. Config the Huggingface token and root path in python files  
4. If using Slurm, also config the batch file  
5. Put the corresponding csv file in the root folder if needed  
6. Use the train-array.sh for fine-tuning (Slurm), or run the train-mistral7b.py with parameter (from 0 to 47, 48 different combination of hyperparameters; 25 gives the best result)  