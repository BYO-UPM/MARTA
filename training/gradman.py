import torch
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
from training.min_norm_solvers import MinNormSolver


OUT_FILE_MAG = 'local_results/trace_mag.csv'
OUT_FILE_COS = 'local_results/trace_cos.csv'
OUT_FILE_LOSS = 'local_results/trace_loss.csv'


def get_tensor_from_model(model, task_count, excluded_layers):
    grad_dims = []
    for name, param in model.named_parameters():
        # Exclude those layers after the GMVAE
        if name.split('.')[0] not in excluded_layers:
            grad_dims.append(param.data.numel())
    grads = torch.Tensor(sum(grad_dims), task_count).cuda()
    return grads, grad_dims


def grad2vec(m, grads, grad_dims, task):
    # store the gradients
    grads[:, task].fill_(0.0)
    cnt = 0
    for name, param in m.named_parameters():
        # Exclude those layers after the GMVAE
        if name.split('.')[0] not in ['spec_dec', 'clf_cnn', 'clf_mlp', 'hmc']:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1


def overwrite_grad(m, newgrad, grad_dims, task_count, excluded_layers):
    newgrad = newgrad * task_count # to match the sum loss
    cnt = 0
    for name, param in m.named_parameters():
        # Exclude those layers after the GMVAE
        if name.split('.')[0] not in excluded_layers:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone().to(param.device)
            cnt += 1


def graddrop(grads):
    P = 0.5 * (1. + grads.sum(1) / (grads.abs().sum(1)+1e-8))
    U = torch.rand_like(grads[:,0])
    M = P.gt(U).view(-1,1)*grads.gt(0) + P.lt(U).view(-1,1)*grads.lt(0)
    g = (grads * M.float()).mean(1)
    return g

def mgd(grads):
    grads_cpu = grads.t().cpu()
    sol, min_norm = MinNormSolver.find_min_norm_element([
        grads_cpu[t] for t in range(grads.shape[-1])])
    w = torch.FloatTensor(sol).to(grads.device)
    g = grads.mm(w.view(-1, 1)).view(-1)
    return g

def pcgrad(grads, rng, task_count):
    grad_vec = grads.t()
    num_tasks = task_count

    shuffled_task_indices = np.zeros((num_tasks, num_tasks - 1), dtype=int)
    for i in range(num_tasks):
        task_indices = np.arange(num_tasks)
        task_indices[i] = task_indices[-1]
        shuffled_task_indices[i] = task_indices[:-1]
        rng.shuffle(shuffled_task_indices[i])
    shuffled_task_indices = shuffled_task_indices.T

    normalized_grad_vec = grad_vec / (
        grad_vec.norm(dim=1, keepdim=True) + 1e-8
    )  # num_tasks x dim
    modified_grad_vec = deepcopy(grad_vec)
    for task_indices in shuffled_task_indices:
        normalized_shuffled_grad = normalized_grad_vec[
            task_indices
        ]  # num_tasks x dim
        dot = (modified_grad_vec * normalized_shuffled_grad).sum(
            dim=1, keepdim=True
        )  # num_tasks x dim
        modified_grad_vec -= torch.clamp_max(dot, 0) * normalized_shuffled_grad
    g = modified_grad_vec.mean(dim=0)
    return g


def cagrad(grads, task_count, alpha=0.5, rescale=1):
    GG = grads.t().mm(grads).cpu() # [num_tasks, num_tasks]
    g0_norm = (GG.mean()+1e-8).sqrt() # norm of the average gradient

    x_start = np.ones(task_count) / task_count
    bnds = tuple((0,1) for x in x_start)
    cons=({'type':'eq','fun':lambda x:1-sum(x)})
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha*g0_norm+1e-8).item()
    def objfn(x):
        return (x.reshape(1,task_count).dot(A).dot(b.reshape(task_count, 1)) + \
                c * np.sqrt(x.reshape(1,task_count).dot(A).dot(x.reshape(task_count,1))+1e-8)).sum()
    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grads.device)
    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm+1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale== 0:
        return g
    elif rescale == 1:
        return g / (1+alpha**2)
    else:
        return g / (1 + alpha)
    

def monitor_grad(epoch, batch, grads, task_count):
    
    if (batch == 0) and (epoch == 0): 

        with open(OUT_FILE_MAG, 'w') as f_mag:
            f_mag.write(f'epoch,batch,idx,mag\n')
        with open(OUT_FILE_COS, 'w') as f_cos:
            f_cos.write(f'epoch,batch,idx_i,idx_j,cos\n')

    else:

        grad_mag = []

        # Dump gradients magnitude to file
        with open(OUT_FILE_MAG, 'a') as f_mag:
            for i in range(task_count):
                mag_i = torch.norm(grads[:,i])
                grad_mag.append(mag_i)
                f_mag.write(f'{epoch},{batch},{i},{mag_i:.4f}\n')
        
        # Dump gradients cosine to file
        with open(OUT_FILE_COS, 'a') as f_cos:
            for i in range(task_count):
                for j in range(i+1, task_count):
                    cos_ij = torch.dot(grads[:,i], grads[:,j]) / grad_mag[i] / grad_mag[j]
                    f_cos.write(f'{epoch},{batch},{i},{j},{cos_ij:.6f}\n')


def monitor_loss(epoch, batch, losses, task_count):
    
    if (batch == 0) and (epoch == 0): 

        with open(OUT_FILE_LOSS, 'w') as f_loss:
            f_loss.write(f'epoch,batch,idx,loss\n')

    else:

        # Dump task loss magnitude to file
        with open(OUT_FILE_LOSS, 'a') as f_loss:
            for i in range(task_count):
                f_loss.write(f'{epoch},{batch},{i},{losses[i]:.4f}\n')

