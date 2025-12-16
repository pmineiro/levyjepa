import torch
import torch.nn as nn
import torch.nn.functional as F

class Stencil(nn.Module):
    """
    Extracts independent Lévy increments from a batch of images
    using either the axis-aligned stencil or the diagonal (45°) stencil.

    Input:  (B, W, H, D)
    Output: (B, N, D)
    """
    def __init__(self, *, axis_aligned: bool):
        super().__init__()
        self.axis_aligned = axis_aligned
        # we will create kernels when forward is called because D is dynamic.

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: (B, W, H, D) — note spatial dims are W, H in your code
        B, W, H, D = Z.shape
        # conv expects (B, C, H, W). We'll treat channels=C=D
        x = Z.permute(0, 3, 2, 1).contiguous()  # -> (B, D, H, W)
        # Note: permute to (B, C, H, W) and ensure contiguous for conv

        device = x.device
        dtype = x.dtype

        if self.axis_aligned:
            # kernel for second-order rectangle difference:
            #  Z(x,y) - Z(x,y-1) - Z(x-1,y) + Z(x-1,y-1)
            # as conv kernel with origin at bottom-right of 2x2 block:
            # kernel (1,1) = +1, (1,0) = -1, (0,1) = -1, (0,0) = +1
            k = torch.tensor([[+1., -1.],
                              [-1., +1.]], device=device, dtype=dtype)
        else:
            # diagonal stencil:
            # A(x-1, y) - A(x, y-1) - A(x, y+1) + A(x+1, y)
            # kernel of size 3x3 with center at (1,1) (0-indexed)
            k = torch.zeros((3, 3), device=device, dtype=dtype)
            # positions:
            # (0,1) = A(x-1,y)  -> +1
            # (1,0) = A(x, y-1) -> -1
            # (1,2) = A(x, y+1) -> -1
            # (2,1) = A(x+1,y)  -> +1
            k[0, 1] = -1.
            k[1, 0] = +1.
            k[1, 2] = +1.
            k[2, 1] = -1.

        # Expand kernel to grouped convs: shape (D, 1, Kh, Kw)
        Kh, Kw = k.shape
        kernel = k.view(1, 1, Kh, Kw).repeat(D, 1, 1, 1)  # grouped conv, one kernel per channel

        # Perform grouped conv: in_channels = out_channels = D, groups=D
        out = F.conv2d(x, weight=kernel, bias=None, stride=1, padding=0, groups=D)

        # out shape: (B, D, H_out, W_out)
        # we want (B, N, D) where N = H_out * W_out
        out = out.permute(0, 3, 2, 1).contiguous()  # -> (B, H_out, W_out, D)
        return out.reshape(B, -1, D)

if __name__ == "__main__":
    from tqdm import tqdm

    class OldStencil(nn.Module):
        def __init__(self, *, axis_aligned: bool):
            super().__init__()
            self.axis_aligned = axis_aligned

        def forward(self, Z: torch.Tensor) -> torch.Tensor:
            """
            Z: (B, W, H, D)

            Returns: increments of shape (B, N, D)
            """
            B, W, H, D = Z.shape

            if self.axis_aligned:
                # Standard second-order rectangular difference
                #
                # ΔZ(x,y) = Z(x,y) - Z(x,y-1) - Z(x-1,y) + Z(x-1,y-1)
                #
                # Valid stencil points require:
                #   x >= 1, y >= 1
                # We compute only those.
                Z_xy     = Z[:, 1:W,   1:H,   :]   # (B, W-1, H-1, D)
                Z_x_y1   = Z[:, 1:W,   0:H-1, :]   # (B, W-1, H-1, D)
                Z_x1_y   = Z[:, 0:W-1, 1:H,   :]   # (B, W-1, H-1, D)
                Z_x1_y1  = Z[:, 0:W-1, 0:H-1, :]   # (B, W-1, H-1, D)

                delta = Z_xy - Z_x_y1 - Z_x1_y + Z_x1_y1   # (B, W-1, H-1, D)

            else:
                # Diagonal ("southern cone") difference
                #
                # ΔA(x,y) = A(x-1, y) - A(x, y-1) - A(x, y+1) + A(x+1, y)
                #
                # Valid stencil points require:
                #   1 <= x <= W-2, 1 <= y <= H-2
                # because we need x-1 >= 0, x+1 < W and y-1 >= 0, y+1 < H.
                A_x1_y  = Z[:, 0:W-2, 1:H-1, :]   # A(x-1, y)
                A_x_y1  = Z[:, 1:W-1, 0:H-2, :]   # A(x, y-1)
                A_x_y1p = Z[:, 1:W-1, 2:H,   :]   # A(x, y+1)
                A_x1p_y = Z[:, 2:W,   1:H-1, :]   # A(x+1, y)

                delta = A_x1_y - A_x_y1 - A_x_y1p + A_x1p_y   # (B, W-2, H-2, D)

            return delta.reshape(B, -1, D)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    def generate_test_cases(total):
        for _ in range(total):
            axis_aligned = torch.randint(0, 2, (1,)).item() == 1
            B = torch.randint(1, 5, (1,)).item()
            D = torch.randint(1, 7, (1,)).item()

            min_size = 2 if axis_aligned else 3
            W = torch.randint(min_size, 15, (1,)).item()
            H = torch.randint(min_size, 15, (1,)).item()

            yield axis_aligned, B, W, H, D

    for axis_aligned, B, W, H, D in tqdm(generate_test_cases(1000), total=1000):
        new_op = Stencil(axis_aligned=axis_aligned).to(device)
        old_op = OldStencil(axis_aligned=axis_aligned).to(device)

        all_match = True

        Z = torch.randn(B, W, H, D, device=device, dtype=torch.float32) * 10

        with torch.no_grad():
            out_new = new_op(Z)
            out_old = old_op(Z)

        # Compare shapes first
        if out_new.shape != out_old.shape:
            raise ValueError(f"  Case {axis_aligned, B, W, H, D} SHAPE MISMATCH: new {out_new.shape}, old {out_old.shape}")

        # Compare values
        atol = 1e-5
        if not torch.allclose(out_new, out_old, atol=atol):
            max_diff = torch.max(torch.abs(out_new - out_old))
            mean_diff = torch.mean(torch.abs(out_new - out_old))
            raise ValueError(f"  Case {axis_aligned, B, W, H, D} FAIL! max diff: {max_diff:.2e}, mean diff: {mean_diff:.2e}:\nNew output: {out_new.flatten()[:20]}\nOld output: {out_old.flatten()[:20]}")

    print("\nTest completed.")
