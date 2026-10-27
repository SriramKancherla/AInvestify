"""
Standard DDPM (Denoising Diffusion Probabilistic Model) mechanics: the
forward noising process, the training loss, and reverse sampling.

This is the same forward/reverse process described conceptually in the
project's "Background: What Are Diffusion Models?" slide - this module is
the actual implementation of it.
"""

import torch
import torch.nn.functional as F


class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
        out = a.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward process: x0 -> x_t, adding noise according to the schedule."""
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1m_ac = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ac * x0 + sqrt_1m_ac * noise

    def training_loss(
        self,
        model,
        x0: torch.Tensor,
        x_cond: torch.Tensor,
        horizon_idx: torch.Tensor,
        cfg_drop_prob: float = 0.0,
    ) -> torch.Tensor:
        """
        Epsilon-prediction DDPM loss. If cfg_drop_prob > 0, a random subset
        of the batch has x_cond zeroed so the model also learns the
        unconditional score (classifier-free guidance).
        """
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device).long()
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        if cfg_drop_prob > 0:
            drop = torch.rand(b, device=x0.device) < cfg_drop_prob
            x_cond = x_cond.clone()
            x_cond[drop] = 0
        predicted_noise = model(x_t, x_cond, t, horizon_idx)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, x_cond: torch.Tensor, t: torch.Tensor, horizon_idx: torch.Tensor) -> torch.Tensor:
        """One reverse diffusion step: x_t -> x_{t-1}."""
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_1m_ac_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)

        predicted_noise = model(x_t, x_cond, t, horizon_idx)
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_1m_ac_t)

        if (t == 0).all():
            return model_mean
        posterior_var_t = self._extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        x_cond: torch.Tensor,
        horizon_idx: torch.Tensor,
        img_size: int,
        channels: int = 1,
        num_steps: int = 100,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Faster reverse sampling (DDIM). Same trained weights; fewer steps than full DDPM.

        guidance_scale > 1 applies classifier-free guidance (requires a model
        trained with cfg_drop_prob > 0). scale=1 is the unguided draw.
        """
        b = x_cond.shape[0]
        device = x_cond.device
        x = torch.randn(b, channels, img_size, img_size, device=device)

        num_steps = max(2, min(int(num_steps), self.timesteps))
        ts = torch.linspace(self.timesteps - 1, 0, steps=num_steps, device=device).long()

        for i in range(len(ts)):
            t_cur = ts[i]
            t_prev = ts[i + 1] if i + 1 < len(ts) else None
            t_batch = torch.full((b,), int(t_cur.item()), device=device, dtype=torch.long)

            eps = model(x, x_cond, t_batch, horizon_idx)
            if guidance_scale > 1.0:
                eps_u = model(x, torch.zeros_like(x_cond), t_batch, horizon_idx)
                eps = eps_u + guidance_scale * (eps - eps_u)
            a_t = self._extract(self.alphas_cumprod, t_batch, x.shape)
            sqrt_a_t = torch.sqrt(a_t)
            sqrt_1m_a_t = torch.sqrt(1.0 - a_t)
            pred_x0 = (x - sqrt_1m_a_t * eps) / sqrt_a_t
            pred_x0 = pred_x0.clamp(-1.0, 1.0)

            if t_prev is None or int(t_cur.item()) == 0:
                x = pred_x0
                break

            t_prev_batch = torch.full((b,), int(t_prev.item()), device=device, dtype=torch.long)
            a_prev = self._extract(self.alphas_cumprod, t_prev_batch, x.shape)
            sigma = (
                eta
                * torch.sqrt((1.0 - a_prev) / (1.0 - a_t))
                * torch.sqrt(1.0 - a_t / a_prev)
            )
            dir_xt = torch.sqrt((1.0 - a_prev - sigma**2).clamp(min=0.0)) * eps
            noise = torch.randn_like(x) if float(eta) > 0 else 0.0
            x = torch.sqrt(a_prev) * pred_x0 + dir_xt + sigma * noise

        return x

    @torch.no_grad()
    def sample(self, model, x_cond: torch.Tensor, horizon_idx: torch.Tensor, img_size: int, channels: int = 1) -> torch.Tensor:
        """Full reverse process: pure noise -> a generated chart, conditioned
        on x_cond and horizon_idx. This is the actual forecasting step."""
        b = x_cond.shape[0]
        x = torch.randn(b, channels, img_size, img_size, device=x_cond.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=x_cond.device, dtype=torch.long)
            x = self.p_sample(model, x, x_cond, t, horizon_idx)
        return x