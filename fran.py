import torch.nn.functional as F
import torch


def downsample(x, step=GRAIN):
    down = torch.zeros([len(x), 3, DATA_SHAPE//step, DATA_SHAPE//step])

    for i in range(0, DATA_SHAPE, step):
        for j in range(0, DATA_SHAPE, step):
            v = x[:, :, i:i+step, j:j+step].mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            ii, jj = i // step, j // step
            down[:, :, ii:ii+1, jj:jj+1] = v
    return down

def upsample(x, step=GRAIN):
    up = torch.zeros([len(x), 3, DATA_SHAPE, DATA_SHAPE])

    for i in range(0, DATA_SHAPE, step):
        for j in range(0, DATA_SHAPE, step):
            ii, jj = i // step, j // step
            up[:, :, i:i+step, j:j+step] = x[:, :, ii:ii+1, jj:jj+1]
    return up


# Compute the gradient of the loss w.r.t. the input data
def gradient_wrt_data(model,device,data,lbl, loss_list, num_im):
    dat = data.clone().detach()
    dat.requires_grad = True
    out = model(dat)[0]
    loss = F.cross_entropy(out,lbl)
    model.zero_grad()
    loss.backward()
    loss_list.append(loss.item()/num_im)
    data_grad = dat.grad.data
    return data_grad.data.detach(), loss_list

def PGD_attack(model, device, dat, lbl, eps, alpha, iters, rand_start, loss_list, iter_list, num_im):
    # TODO: Implement the PGD attack
    # - dat and lbl are tensors
    # - eps and alpha are floats
    # - iters is an integer
    # - rand_start is a bool

    # x_nat is the natural (clean) data batch, we .clone().detach()
    # to copy it and detach it from our computational graph
    x_nat = dat.clone().detach()

    # If rand_start is True, add uniform noise to the sample within [-eps,+eps],
    # else just copy x_nat
    # if rand_start == True:
    #   x_nat_clone = x_nat + torch.FloatTensor(x_nat.shape).uniform_(-eps, eps).to(device)
    # else:
    #   x_nat_clone = x_nat

    # Make sure the sample is projected into original distribution bounds [0,1]
    x_nat_clone = torch.clamp(x_nat.clone().detach(), 0., 1.)
    # x_nat_clone = x_nat

    # Iterate over iters
    for iter in range(iters):
      iter_list.append(iter)

        # Compute gradient w.r.t. data (we give you this function, but understand it)
      gradient, loss_list = gradient_wrt_data(model, device, x_nat_clone, lbl, loss_list, num_im)
      # gradient = generation_loss(model, x_nat_clone, lbl)
      # gradient_flatten = gradient.view(gradient.shape[0],-1)
      # l2_of_grad = torch.norm(gradient_flatten, p=2, dim=1)
      # l2_of_grad = torch.clamp(l2_of_grad, min=1e-12)
      # norm_grad = gradient/l2_of_grad
      l = len(x_nat.shape) - 1
      g_norm = torch.norm(gradient.view(gradient.shape[0], -1), dim=1).view(-1, *([1]*l))
      norm_grad = gradient / (g_norm + 1e-10)
        # Perturb the image using the gradient
      x_nat_clone = x_nat_clone - (alpha * norm_grad)
        # Clip the perturbed datapoints to ensure we still satisfy L_infinity constraint
                  #projection step
      diff = x_nat_clone - x_nat
      diff = diff.renorm(p=2, dim=0, maxnorm=eps)
      x_nat_clone = torch.clamp(x_nat.clone().detach() + diff.clone().detach(), 0., 1.)

        # Clip the perturbed datapoints to ensure we are in bounds [0,1]
      # x_nat_clone = torch.clamp(x_nat_clone.clone().detach(), 0, 1)

    # Return the final perturbed samples
    return x_nat_clone, loss_list, iter_list
