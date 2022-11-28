# GGO attack

The framework of data processing and training is taken from this [repository](https://github.com/Xtra-Computing/NIID-Bench). (Li Q, Diao Y, Chen Q, et al. Federated learning on non-iid data silos: An experimental study. ICDE, 2022: 965-978.)


## Usage
Here is one example to run this code:
```
python experiments.py --model=simple-cnn \
    --dataset=generated \
    --alg=robust-bar \
    --lr=0.01 \
    --batch-size=64 \
    --epochs=5 \
    --n_parties=27 \
    --rho=0.9 \
    --comm_round=20 \
    --partition=noniid-labeldir \
    --beta=0.5\
    --device='cpu'\
    --datadir='./data/' \
    --logdir='./logs/' \
    --noise=0 \
    --init_seed=0 \
    --atk_type='our-with-collusion' \
    --def_type='krum' \
    --n_Byzantine=12 \
    --perturb_weight=1 \
    --threshold=0.5 \
    --tau=1 \
    --L=5 \
    --n_samples=50
```

