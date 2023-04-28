import argparse
import torch
from collections import OrderedDict, defaultdict
import numpy as np
import torch.utils.data
from tqdm import trange
import sys, random, json, os
from utils import get_args, mkdir
from models import CNNHyper, CNNTarget
from dataset import gen_random_loaders

# from utils import get_logger

class LocalTrainer:
    def __init__(self, args, net, device):
        self.train_loaders, self.val_loaders, self.test_loaders = gen_random_loaders(
            args.data_name, args.data_path, args.num_nodes, args.batch_size, args.classes_per_node)

        self.device = device
        self.args = args
        self.net = net
        self.criteria = torch.nn.CrossEntropyLoss()

    def __len__(self):
        return self.n_nodes

    def train(self, weights, client_id):
        self.net.load_state_dict(weights)
        self.net.train()
        inner_state = OrderedDict({k: t.data for k, t in weights.items()})
        optimizer = torch.optim.SGD(
            self.net.parameters(), lr=self.args.inner_lr, momentum=.9, weight_decay=self.args.inner_wd)

        for i in range(self.args.inner_steps):
            
            batch = next(iter(self.train_loaders[client_id]))
            img, label = tuple(t.to(self.device) for t in batch)

            pred = self.net(img)
            loss = self.criteria(pred, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 50)

            optimizer.step()

        final_state = self.net.state_dict()

        return OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})
    
    @torch.no_grad()
    def evalute(self, weights, client_id, split):
        running_loss, running_correct, running_samples = 0., 0., 0.
        if split == 'test':
            eval_data = trainer.test_loaders[client_id]
        elif split == 'val':
            eval_data = trainer.val_loaders[client_id]
        else:
            eval_data = trainer.train_loaders[client_id]
        
        self.net.load_state_dict(weights)

        for x, y in eval_data:
            x = x.to(self.device)
            y = y.to(self.device)
            
            pred = self.net(x)
            running_loss += self.criteria(pred, y).item()
            # print(running_loss)
            running_correct += pred.argmax(1).eq(y).sum().item()
            running_samples += len(y)
        return running_loss/(len(eval_data) + 1), running_correct, running_samples

def evaluate(hnet, trainer, clients, split):
    results = defaultdict(lambda: defaultdict(list))
    hnet.eval()

    for client_id in clients:     
        weights = hnet(torch.tensor([client_id], dtype=torch.long).to(device))
        running_loss, running_correct, running_samples = trainer.evalute(weights, client_id, split)

        results[client_id]['loss'] = running_loss
        results[client_id]['correct'] = running_correct
        results[client_id]['total'] = running_samples
    
    total_correct = sum([val['correct'] for val in results.values()])
    total_samples = sum([val['total'] for val in results.values()])
    avg_loss = np.mean([val['loss'] for val in results.values()])
    avg_acc = total_correct / total_samples

    all_acc = [val['correct'] / val['total'] for val in results.values()]

    return results, avg_loss, avg_acc, all_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    # logger = get_logger(filename='./log.txt', enable_console=True)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if args.cuda == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % args.cuda)

    if args.classes_per_node == 0:
        if args.data_name == 'cifar100':
            args.classes_per_node = 10
        else:
            args.classes_per_node = 2

    if args.data_name == "cifar10":
        hnet = CNNHyper(args.num_nodes, args.n_embeds, args.embed_dim, n_kernels=args.n_kernels, norm_var=args.norm_var,
            layer_wise=args.layer_wise, use_fc=args.use_fc, hdim=args.hdim)
        net = CNNTarget(n_kernels=args.n_kernels)
    elif args.data_name == "cifar100":
        hnet = CNNHyper(args.num_nodes, args.n_embeds, args.embed_dim, n_kernels=args.n_kernels, out_dim=100, norm_var=args.norm_var,
            layer_wise=args.layer_wise, use_fc=args.use_fc, hdim=args.hdim)
        net = CNNTarget(n_kernels=args.n_kernels, out_dim=100)
    elif args.data_name == "mnist":
        hnet = CNNHyper(args.num_nodes, args.n_embeds, args.embed_dim, in_channels=1, n_kernels=args.n_kernels, norm_var=args.norm_var,
            layer_wise=args.layer_wise, use_fc=args.use_fc, hdim=args.hdim)
        net = CNNTarget(n_kernels=args.n_kernels, in_channels=1)
    else:
        raise ValueError("choose data_name from ['cifar10', 'cifar100', 'mnist']")

    # print(args.model_path)
    hnet.load_state_dict(torch.load(args.model_path))

    hnet = hnet.to(device)
    net = net.to(device)

    trainer = LocalTrainer(args, net, device)

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(params=hnet.coeff.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(params=hnet.coeff.parameters(), lr=args.lr)
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(params=hnet.coeff.parameters(), lr=args.lr)
    else:
        print('invalid optim')
        exit()

    # optimizer = optimizers[args.optim]
    criteria = torch.nn.CrossEntropyLoss()

    if args.train_clients == -1 or args.test_last == -1:
        print('(train_clients or test_last) = 1')
        exit()

    train_list = range(args.num_nodes - args.test_last, args.num_nodes)

    results = defaultdict(list)
    for step in trange(args.num_steps):
        hnet.train()

        for client_id in train_list:
            # produce & load local network weights
            weights = hnet(torch.tensor([client_id], dtype=torch.long).to(device))
            net.load_state_dict(weights)

            delta_theta = trainer.train(weights, client_id)

            final_state = net.state_dict()
            
            optimizer.zero_grad()

            hnet_grads = torch.autograd.grad(
                list([w/args.clients_per_round for w in weights.values()]), hnet.parameters(), 
                grad_outputs=[v/args.clients_per_round for v in delta_theta.values()], allow_unused=True)

            # update hnet weights
            for p, g in zip(hnet.parameters(), hnet_grads):
                p.grad = g

            torch.nn.utils.clip_grad_norm_(hnet.parameters(), args.grad_clip)
            optimizer.step()

        if step % args.eval_every == 0:
            last_eval = step
            step_results, avg_loss, avg_acc, all_acc = evaluate(hnet, trainer, train_list, split="test")
            # logging.info(f"\nStep: {step+1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")
            print(f"Step: {step+1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")
            results['test_avg_loss'].append(avg_loss)
            results['test_avg_acc'].append(avg_acc)

            _, val_avg_loss, val_avg_acc, _ = evaluate(hnet, trainer, train_list, split="val")
            results['val_avg_loss'].append(val_avg_loss)
            results['val_avg_acc'].append(val_avg_acc)

    if step != last_eval:
        _, val_avg_loss, val_avg_acc, _ = evaluate(hnet, trainer, train_list, split="val")
        step_results, avg_loss, avg_acc, all_acc = evaluate(hnet, trainer, train_list, split="test")
        # logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")
        print(f"Step: {step+1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")

        results['test_avg_loss'].append(avg_loss)
        results['test_avg_acc'].append(avg_acc)

        results['val_avg_loss'].append(val_avg_loss)
        results['val_avg_acc'].append(val_avg_acc)


    mkdir(args.save_path)
    if args.suffix:
        args.suffix = '_' + args.suffix
    save_title = f"results_cn_{args.num_nodes}_ne_{args.n_embeds}_ed_{args.embed_dim}_gc_{args.grad_clip}{args.suffix}"

    with open(os.path.join(args.save_path, save_title+'.json'), "w") as file:
        json.dump(results, file, indent=4)
    

