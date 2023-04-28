# pFedLHN

----

## Installation

- Create a virtual environment with conda/virtualenv (python>=3.8)
- Clone this repository
- Run: `pip install -r requirements.txt` to install packages

## run experiments

```bash
# results on [cifar10/cifar100/mnist]
# --num-nodes: number of clients
python run.py --data-path ./data --data-name cifar10 --save-path test_cifat10 \
		--num-nodes 50 --num-steps 3000 --cuda 0 --optim adamw --lr 3e-4;
		
# to compare with non-layer-wise result, add argument: --layer-wise False
```

```bash
# generalization on new clients
# first use 40 out of 50 clients to train the model
python run.py --data-path ./data --data-name cifar10 \
    --save-path cifar10_test_new --num-nodes 50 --num-steps 2000 --cuda 0 \
    --optim adamw --lr 3e-4 --suffix tr40 \
    --train-clients 40 --save-model true --random-seed 42;

# then use the rest 10 clients to test model generalization.
# In this phase, several local training steps will be conduct on new clients to produce personalized models
# use same [--random-seed] to ensure train/test clients partition consistent with training phase
python run_test.py --data-path ./data --data-name cifar10 \
    --save-path cifar10_test_new --num-nodes 50 --num-steps 100 --cuda 0 \
    --optim adamw --lr 3e-4 --suffix test10 \
    --train-clients 40 --test-last 10 --eval-every 4 --random-seed 42 \
    --model-path cifar10_test_new/[model.pth];
```
