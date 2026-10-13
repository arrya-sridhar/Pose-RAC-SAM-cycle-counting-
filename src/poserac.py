"""Port of the official PoseRAC (github.com/MiracleDance/PoseRAC) to mask-derived pose vectors.

Kept from the official code:
  * nn.TransformerEncoder(TransformerEncoderLayer(d_model=dim, nhead=heads), 6 layers) + Linear(dim, C) + sigmoid
  * mode='official': x.view(-1, 1, dim) with batch_first=False -> attention runs ACROSS FRAMES
      (random 16-frame batches during training, the whole video at inference) -- exactly what the released code does.
    mode='perframe': each frame is its own sequence (the per-pose model the paper describes).
  * labels: salient-1 -> 1 at the class index, salient-2 -> 0;  BCE on sigmoid outputs
  * loss = alpha * (-TripletMarginLoss with MultiSimilarityMiner) + (1 - alpha) * BCE, alpha = 0.01 (sign as released)
  * Adam lr 1e-3, batch 16, ReduceLROnPlateau(factor .75, patience 6), val set = train set, early stopping
  * counting: EMA momentum 0.4, Action_trigger(enter 0.78, exit 0.4) on p and 1-p, count salient alternations
Changed (necessary for the robot / tiny label sets):
  * input = 99-d mask pose (grid or polar) instead of BlazePose; no per-frame min-max normalisation
  * minimum number of optimisation steps (20 epochs over ~20 labelled frames is only ~40 steps)
"""
import numpy as np, torch, torch.nn as nn
from pytorch_metric_learning import miners, losses


class PoseRACNet(nn.Module):
    def __init__(self, dim=99, heads=9, enc_layer=6, num_classes=1, mode='official'):
        super().__init__()
        self.dim, self.mode = dim, mode
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=heads),
                                                         num_layers=enc_layer, enable_nested_tensor=False)
        self.fc1 = nn.Linear(dim, num_classes)

    def embed(self, x):
        x = x.view(-1, 1, self.dim) if self.mode == 'official' else x.view(1, -1, self.dim)
        x = self.transformer_encoder(x)
        return x.reshape(-1, self.dim)

    def forward(self, x):
        return self.fc1(self.embed(x))


def train_poserac(X, Y, dim, heads, mode='official', lr=1e-3, alpha=0.01, epochs=20, min_steps=300, seed=42,
                  batch=16, patience=20):
    torch.manual_seed(seed); np.random.seed(seed)
    net = PoseRACNet(dim, heads, 6, Y.shape[1], mode)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.75, patience=6, mode='min', min_lr=10e-7)
    bce = nn.BCELoss(); miner = miners.MultiSimilarityMiner(); trip = losses.TripletMarginLoss()
    Xt = torch.tensor(X, dtype=torch.float32); Yt = torch.tensor(Y, dtype=torch.float32)
    n = len(Xt); spe = int(np.ceil(n / batch))
    epochs = max(epochs, int(np.ceil(min_steps / spe)))
    best, bad, best_state = 1e9, 0, None
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        for b in range(spe):
            idx = perm[b * batch:(b + 1) * batch]
            x, y = Xt[idx], Yt[idx]
            emb = net.embed(x)
            lab = torch.argmax(y, dim=1)
            pairs = miner(emb, lab)
            l_metric = -trip(emb, lab, pairs)
            l_cls = bce(torch.sigmoid(net.fc1(emb)), y)
            loss = alpha * l_metric + (1 - alpha) * l_cls
            opt.zero_grad(); loss.backward(); opt.step()
        # validation on the training set, as in the official train.py (batched by 16 like its val loader)
        net.eval()
        with torch.no_grad():
            vl = np.mean([bce(torch.sigmoid(net(Xt[i:i + batch])), Yt[i:i + batch]).item() for i in range(0, n, batch)])
        sched.step(vl)
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


def predict(net, X, cls=0):
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(X, dtype=torch.float32)))[:, cls].numpy()


class ActionTrigger:
    """verbatim logic of the official Action_trigger"""

    def __init__(self, enter_threshold=0.78, exit_threshold=0.4):
        self.enter, self.exit, self.entered = enter_threshold, exit_threshold, False

    def __call__(self, s):
        if not self.entered:
            self.entered = s > self.enter
            return False
        if s < self.exit:
            self.entered = False
            return True
        return False


def count_official(p, enter=0.78, exit=0.4, momentum=0.4):
    """official eval.py counting loop.  Returns count, count-event frame indices, home-entry frame indices."""
    r1, r2 = ActionTrigger(enter, exit), ActionTrigger(enter, exit)
    prob, count, curr, init = 0.5, 0, 'holder', 'pose_holder'
    events, home_entries = [], []
    was_in1 = False
    for i, o in enumerate(p):
        prob = o * (1. - momentum) + momentum * prob
        s1 = r1(prob); s2 = r2(1 - prob)
        if init == 'pose_holder':
            if s1: init = 'salient1'
            elif s2: init = 'salient2'
        if init == 'salient1':
            if curr == 'salient1' and s2:
                count += 1; events.append(i)
        else:
            if curr == 'salient2' and s1:
                count += 1; events.append(i)
        if s1: curr = 'salient1'
        elif s2: curr = 'salient2'
        # boundary proxy: the moment the smoothed home-probability enters salient-1 after a salient-2 excursion
        if r1.entered and not was_in1 and curr == 'salient2':
            home_entries.append(i)
        was_in1 = r1.entered
    return count, events, home_entries
