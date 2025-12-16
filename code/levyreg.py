import numbers
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import brentq
from scipy.special import roots_laguerre
from typing import Callable, TypeVar
from validatekwargs import validate_kwargs

T = TypeVar('T')

class Levy2DRegularizer(nn.Module):
    """
    Epps–Pulley-style regularizer for Lévy sheet increments (projected to 1D).

    Characteristic function used:
        ϕ(u) = exp( -σ² u² / 2 + λ (α² / (α² + u²) - 1) )

    This corresponds to:
      - Brownian sheet component with volatility σ
      - Compound Poisson jumps with intensity λ
      - Jump sizes follow:  y = √τ · z
        where  z ∼ 𝒩(0, 1)   and   τ ∼ Exp( rate = α² / 2 )

    The loss is:
        loss_b = N * ∫ |ϕ̂(u) - ϕ(u)|² w(u) du    for each batch element b
    where N = number of increments
    """
    @validate_kwargs(alpha=(numbers.Number, lambda x: x > 0),
                     sigma=(numbers.Number, lambda x: x > 0),
                     lam=(numbers.Number, lambda x: x > 0),
                     n_points=(int, lambda x: x > 10),
                    )
    def __init__(
        self,
        alpha: float,
        sigma: float,
        lam: float,                     # λ (jump intensity)
        n_points: int = 17
    ):
        super().__init__()

        with torch.no_grad():
            # Choose u_max so that theoretical characteristic function ϕ(u) is small for cell area 1
            # we take the right hand side of the bracketed root
            u_max = Levy2DRegularizer.find_root(lambda z: 4 + Levy2DRegularizer.log_theoretical_cf(z, alpha=alpha, sigma=sigma, lam=lam))

            # Weight function w(u) - exponential decay is a standard & stable choice
            # ∫_{-∞}^∞ exp(-c|u|) du = 2/c < ∞, and it's symmetric
            c_np = np.float64(10 / u_max)

            print(f'{u_max=} {c_np=} {n_points=}')

            # Gauss-Laguerre nodes/weights for weight e^{-x} on [0, ∞)
            x_np, w_np = roots_laguerre(n_points)

            u = torch.from_numpy(x_np / c_np)
            w = torch.from_numpy(w_np / c_np)

            # Precompute theoretical characteristic function ϕ(u) for cell area 1
            log_phi_theory = self.log_theoretical_cf(u, alpha=alpha, sigma=sigma, lam=lam)

            self.register_buffer("u", u)
            self.register_buffer("w", w)
            self.register_buffer("log_phi_theory", log_phi_theory)

    @staticmethod
    def find_root(fn: Callable[[float], float]) -> tuple[float, float]:
        # Brent's method, assuming: fn(0) > 0 and lim_{x→∞} fn(x) = -∞.

        lb = 0
        assert fn(lb) > 0
        ub = 1
        while fn(ub) > 0:
            ub *= 2
            assert ub < 1e8

        return brentq(fn, 0, ub, xtol=1e-6, rtol=1e-6)

    @staticmethod
    @validate_kwargs(alpha=(numbers.Number, lambda x: x > 0),
                     sigma=(numbers.Number, lambda x: x > 0),
                     lam=(numbers.Number, lambda x: x > 0),
                    )
    def log_theoretical_cf(z: T, *, alpha: float, sigma: float, lam: float) -> T:  # for unit cell area
        sigma2_z2 = 0.5 * sigma**2 * z**2
        jump_part = alpha**2 / (alpha**2 + z**2) - 1.0
        log_phi = -sigma2_z2 + lam * jump_part
        return log_phi

    @validate_kwargs(cell_area=(numbers.Number, lambda x: x > 0))
    def forward(self, increments: torch.Tensor, *, cell_area: float) -> torch.Tensor:
        """
        Compute the Epps–Pulley statistic for a batch of projected increments.

        Args:
            increments: Tensor of shape (B, N) of independent increments
            cell_area:  float, scaling factor for the log characteristic function (default: 1)

        Returns:
            loss: Tensor of shape (B,) - one regularizer value per batch element
        """
        B, N = increments.shape

        u = self.u.to(increments)
        w = self.w.to(increments)

        # Compute theoretical phi with cell_area scaling
        # Cast to phi_hat dtype to have the same floating-point resolution
        with torch.no_grad():
            phi_th = torch.exp(cell_area * self.log_phi_theory).to(increments) # (npoints,)

        # Angles used in the empirical characteristic function
        angle = u.view(-1,1,1) * increments.unsqueeze(0) # (n_points, B, N)

        # |ϕ̂(u) - ϕ(u)|² weighted
        sq_diff = (angle.cos().mean(dim=-1) - phi_th.unsqueeze(1)).square() + angle.sin().mean(dim=-1).square()  # (n_points, B)
        weighted = sq_diff * w.unsqueeze(1)                                                                      # (n_points, B)

        # Integrate: ∫ |...|² w(u) du
        integral = 2.0 * weighted.sum(dim=0)

        return integral

if __name__ == "__main__":
    # Basic smoke tests
    print("Running basic sanity checks for Levy2DRegularizer...")

    # Create a tiny batch of fake increments
    B = 3         # batch size
    N = 50        # number of increments
    torch.manual_seed(0)
    increments = torch.randn(B, N)

    try:
        # Instantiate regularizer with valid parameters
        reg = Levy2DRegularizer(
            alpha=1.5,
            sigma=0.7,
            lam=0.3,
        )
        print("✓ Instantiation succeeded")

        # Forward pass
        loss = reg(increments, cell_area=1.0)

        # Basic checks on output
        assert loss.shape == (B,), f"Loss has wrong shape: {loss.shape}"
        assert torch.isfinite(loss).all(), "Loss contains non-finite values"

        print("✓ Forward pass succeeded")
        print("Loss values:", loss)

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        raise
    else:
        print("All sanity checks passed.")
