## Environments
python 3.7
```bash
pip3 install numpy tensorflow==2.0.0 tqdm
```

## How to run the programs
```bash
python3 hw2.0.py [-T] [-s TESTING_FILE] <TRAINING_DATA> <MODEL_PATH>
python3 hw2.1-1.py [-T] [-s TESTING_FILE] hw2-1_corpus.txt <MODEL_PATH>
python3 hw2.1-2.py [-T] [-s TESTING_FILE] hw2-1_corpus.txt <MODEL_PATH>
```

Options:
* -T, --no-training				Do not train the model
* -s, --submit TESTING_FILE		predict `TESTING_FILE` and output `output.txt`
