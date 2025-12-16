import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import timm, wandb, hydra, tqdm
from timm.layers import AttentionPool2d
from omegaconf import DictConfig
from datasets import load_dataset
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP
from torch.utils.tensorboard import SummaryWriter

from math import sqrt
from levyreg import Levy2DRegularizer
from stencils import Stencil

class WithinImageReg(torch.nn.Module):
    def __init__(self, *, alpha, sigma, lam):
        super().__init__()
        self.axis_aligned_stencil = Stencil(axis_aligned=True)
        self.diagonal_stencil = Stencil(axis_aligned=False)
        self.levy2d = Levy2DRegularizer(alpha=alpha, sigma=sigma, lam=lam)

    # X should be the embedding layer without the CLS token
    def forward(self, X):
        B, W, H, D = X.shape

        A = torch.randn(D, D, device=X.device)
        Q, _ = torch.linalg.qr(A)
        XQ = X @ Q

        XQaa = self.axis_aligned_stencil(XQ[:,:,:,::2])   # shape (B, W, H, D // 2)
        XQdiag = self.diagonal_stencil(XQ[:,:,:,1::2])    # shape (B, W, H, D // 2)

        XQaaloss = self.levy2d(XQaa.view(B, -1), cell_area=1)          # shape (B, )
        XQdiagloss = self.levy2d(XQdiag.view(B, -1), cell_area=2)      # shape (B, )

        return 0.5 * (XQaaloss.mean() + XQdiagloss.mean())

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

class TimmAttentionPoolHead(nn.Module):
    def __init__(self, *, embed_dim, h_patch, w_patch, num_heads, dropout=0.0):
        super().__init__()
        # Reshape patches to spatial: for 128x128 img, patch8 -> 16x16 grid
        self.pooler = AttentionPool2d(
            in_features=embed_dim,
            feat_size=(h_patch, w_patch),
            num_heads=num_heads,
            drop_rate=dropout,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.h_patch = h_patch
        self.w_patch = w_patch

    def forward(self, features):
        B, N, D = features.shape                                                            # (B, 1 + H*W, D)
        patches = features[:, 1:]                                                           # (B, H*W, D)
        patches = patches.transpose(1, 2).reshape(B, D, self.h_patch, self.w_patch)         # (B, D, H, W)
        pooled = self.pooler(patches)                                                       # (B, D)
        pooled = self.norm(pooled)                                                          # (B, D)
        return pooled

class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=128, num_classes=512):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=0,
            drop_path_rate=0.1,
            img_size=128,
        )
        self.patch_size_W = self.backbone.patch_embed.patch_size[0]
        self.patch_size_H = self.backbone.patch_embed.patch_size[1]
        embed_dim = self.backbone.embed_dim
        self.att_patches = TimmAttentionPoolHead(embed_dim=embed_dim,
                                                 h_patch=(128 // self.patch_size_H),
                                                 w_patch=(128 // self.patch_size_W),
                                                 num_heads=6)
        self.logits = torch.nn.Linear(embed_dim, num_classes)
        self.emb_proj = MLP(num_classes, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
        self.patch_proj = MLP(embed_dim, [2048, 2048, proj_dim], norm_layer=nn.LayerNorm)

    def forward(self, x):
        N, V, C, H_img, W_img = x.shape
        assert H_img % self.patch_size_H == 0
        assert W_img % self.patch_size_W == 0
        assert H_img == 128
        assert W_img == 128

        H = H_img // self.patch_size_H
        W = W_img // self.patch_size_W

        full_sequence = self.backbone.forward_features(x.flatten(0, 1)) # (N*V, 1 + H*W, D)
        att_patches = self.att_patches(full_sequence)                   # (N*V, D)
        emb = self.logits(att_patches)                                  # (N*V, num_classes)
        proj_emb = self.emb_proj(emb).reshape(N, V, -1).transpose(0, 1) # (V, N, proj_dim)


        patch_emb = full_sequence[:,1:].reshape(N*V, H, W, -1)          # (N*V, H, W, D)
        patch_proj_emb = self.patch_proj(patch_emb)                     # (N*V, H, W, proj_dim)

        return emb, proj_emb, patch_proj_emb

class HFDataset(torch.utils.data.Dataset):
    def __init__(self, split, V=1):
        self.V = V
        self.ds = load_dataset("frgfm/imagenette", "160px", split=split)
        self.aug = v2.Compose(
            [
                v2.RandomResizedCrop(128, scale=(0.08, 1.0)),
                v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
                v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.test = v2.Compose(
            [
                v2.Resize(128),
                v2.CenterCrop(128),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["image"].convert("RGB")
        transform = self.aug if self.V > 1 else self.test
        return torch.stack([transform(img) for _ in range(self.V)]), item["label"]

    def __len__(self):
        return len(self.ds)


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    wandb.init(project="LeJEPA", config=dict(cfg), sync_tensorboard=True)
    tb_writer = SummaryWriter()
    torch.manual_seed(0)

    device = torch.device(cfg.device_name)

    train_ds = HFDataset("train", V=cfg.V)
    test_ds = HFDataset("validation", V=1)
    train = DataLoader(
        train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=8
    )
    test = DataLoader(test_ds, batch_size=cfg.bs, num_workers=8)

    # modules and loss
    net = ViTEncoder(proj_dim=cfg.proj_dim).to(device)
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 100)).to(device)
    sigreg = SIGReg().to(device)
    withinimagereg = WithinImageReg(alpha=cfg.levy_alpha, sigma=cfg.levy_sigma, lam=cfg.levy_lam).to(device)

    # Optimizer and scheduler
    net_lr = cfg.net_lr_bs256 * sqrt(cfg.bs / 256)
    probe_lr = 1e-3 * sqrt(cfg.bs / 256)
    g1 = {"params": net.parameters(), "lr": net_lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": probe_lr, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])
    warmup_steps = len(train)
    total_steps = len(train) * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=min(net_lr, probe_lr) / 4)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler(enabled=device.type=="cuda")
    global_step = 0
    # Training
    for epoch in range(cfg.epochs):
        net.train(), probe.train()
        for vs, y in tqdm.tqdm(train, total=len(train)):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type=="cuda"):
                vs = vs.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                emb, proj, patch_emb = net(vs)
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                within_loss = withinimagereg(patch_emb)
                orig_lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
                lejepa_loss = within_loss * cfg.gamma + sigreg_loss * cfg.lamb + cfg.kappa * inv_loss * (1 - cfg.lamb)
                y_rep, yhat = y.repeat_interleave(cfg.V), probe(emb.detach())
                probe_loss = F.cross_entropy(yhat, y_rep)
                loss = lejepa_loss + probe_loss
                global_step += 1

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            tb_writer.add_scalar("train/lr", opt.param_groups[0]["lr"], global_step=global_step)
            tb_writer.add_scalar("train/probe_loss", probe_loss.item(), global_step=global_step)
            tb_writer.add_scalar("train/orig_lejepa_loss", orig_lejepa_loss.item(), global_step=global_step)
            tb_writer.add_scalar("train/lejepa_loss", lejepa_loss.item(), global_step=global_step)
            tb_writer.add_scalar("train/sigreg", sigreg_loss.item(), global_step=global_step)
            tb_writer.add_scalar("train/within", within_loss.item(), global_step=global_step)
            tb_writer.add_scalar("train/inv", inv_loss.item(), global_step=global_step)

        # Evaluation
        net.eval(), probe.eval()
        correct = 0
        with torch.inference_mode():
            for vs, y in test:
                vs = vs.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type=="cuda"):
                    correct += (probe(net(vs)[0]).argmax(1) == y).sum().item()
        test_acc = correct / len(test_ds)
        tb_writer.add_scalar("test/acc", test_acc, global_step=epoch+1)
    tb_writer.close()
    wandb.finish()


if __name__ == "__main__":
    main()
