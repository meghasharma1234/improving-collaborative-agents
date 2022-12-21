
### Installation
```
conda create -n cs7648 python=3.8
conda activate cs7648
conda install pip
git clone https://github.com/Stanford-ILIAD/PantheonRL.git
cd PantheonRL
pip install -e .
git submodule update --init --recursive
pip install -e overcookedgym/human_aware_rl/overcooked_ai
pip install imitation
```

### Retieving the human data
```
python save_human_data.py --in_path "PantheonRL/overcookedgym/human_aware_rl/human_aware_rl/data/human/anonymized/clean_main_trials.pkl" --out_path "data/human_data.csv"
```
