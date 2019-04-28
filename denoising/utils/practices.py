import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def annealing_linear(start, end, pct):
    return start + pct * (end-start)


def annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start-end)/2 * cos_out


class OneCycleScheduler(object):
    """
    (0, pct_start) -- linearly increase lr
    (pct_start, 1) -- cos annealing
    """
    def __init__(self, lr_max, div_factor=25., pct_start=0.3):
        super(OneCycleScheduler, self).__init__()
        self.lr_max = lr_max
        self.div_factor = div_factor
        self.pct_start = pct_start
        self.lr_low = self.lr_max / self.div_factor
    
    def step(self, pct):
        # pct: [0, 1]
        if pct <= self.pct_start:
            return annealing_linear(self.lr_low, self.lr_max, pct / self.pct_start)

        else:
            return annealing_cos(self.lr_max, self.lr_low / 1e4, (
                pct - self.pct_start) / (1 - self.pct_start))


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def find_lr(net, trn_loader, optimizer, loss_fn, init_value=1e-8, final_value=10., beta=0.98, device='cuda:1'):
    # https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    num = len(trn_loader) - 1
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for (noisy, clean, _) in trn_loader:
        batch_num += 1
        noisy, clean = noisy.to(device), clean.to(device)
        noisy = noisy.view(-1, *noisy.shape[2:])
        clean = clean.view(-1, *clean.shape[2:])

        #As before, get the loss for this mini-batch of inputs/outputs
        optimizer.zero_grad()
        output = net(noisy)
        loss = loss_fn(output, clean)
        # loss = criterion(outputs, labels)
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(np.log10(lr))
        #Do the SGD step
        loss.backward()
        optimizer.step()
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    print('finished find lr')
    return log_lrs, losses   




if __name__ == '__main__':

    scheduler = OneCycleScheduler(lr_max=0.0005, div_factor=25., pct_start=0.3)

    max_iters = 200 * (5000 // 16)
    pcts = np.arange(max_iters) / max_iters
    lrs = [scheduler.step(pct) for pct in pcts]

    plt.plot(np.arange(max_iters), lrs)
    plt.savefig('one_cycle.png')
    plt.close()


