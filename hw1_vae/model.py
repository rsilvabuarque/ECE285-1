"""The encoder, reparameterization trick, decoder, and lecture-style VAE loss."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConvVAE(nn.Module):
    """Map a grayscale image to a diagonal Gaussian, then back to an image.

    Four stride-two convolutions halve the spatial dimensions four times.
    For 64x64 input: 1x64x64 -> 32x32x32 -> 64x16x16 -> 128x8x8
    -> 256x4x4. The decoder mirrors all four stages, including the last one.
    Normalization layers are omitted to keep this baseline simple and to
    allow a final batch containing just one image.
    """

    def __init__(self, image_size: int = 64, latent_dim: int = 128,
                 base_channels: int = 32):
        super().__init__()
        if image_size < 16 or image_size % 16:
            raise ValueError("image_size must be a positive multiple of 16 (default: 64).")
        if latent_dim < 1 or base_channels < 1:
            raise ValueError("latent_dim and base_channels must be positive.")
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.spatial_size = image_size // 16
        channels = [base_channels * m for m in (1, 2, 4, 8)]

        encoder = []
        in_channels = 1
        for out_channels in channels:
            encoder.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder)
        flat_size = channels[-1] * self.spatial_size ** 2
        # These are different learned heads, not mean/variance of the image batch.
        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_logvar = nn.Linear(flat_size, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, flat_size)

        decoder = []
        for in_channels, out_channels in zip(channels[::-1], channels[-2::-1]):
            decoder.extend([
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        decoder.extend([nn.ConvTranspose2d(channels[0], 1, 4, 2, 1), nn.Sigmoid()])
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        expected = (1, self.image_size, self.image_size)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected [batch, {expected}], received {tuple(x.shape)}.")
        features = self.encoder(x).flatten(start_dim=1)
        # log(variance) may take any real value; exp(logvar) is always positive.
        return self.fc_mu(features), self.fc_logvar(features)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(self, z: Tensor) -> Tensor:
        features = self.decoder_input(z)
        features = features.reshape(-1, self.base_channels * 8,
                                    self.spatial_size, self.spatial_size)
        return self.decoder(features)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(reconstruction: Tensor, target: Tensor, mu: Tensor,
             logvar: Tensor, kl_weight: float = 0.001) -> dict[str, Tensor]:
    """MSE averaged over pixels + weighted KL summed over latent coordinates.

    Matches the practical loss displayed at about 66:00 in the supplied lecture.
    KL is measured in nats per image. This weighted training objective must not
    be labeled an unscaled, unit-weight negative ELBO. Changing MSE from mean
    to sum changes the relative KL strength by the number of image pixels.
    """
    if reconstruction.shape != target.shape:
        raise ValueError("Reconstruction and input must have identical shapes.")
    if kl_weight < 0:
        raise ValueError("kl_weight cannot be negative.")
    reconstruction_loss = F.mse_loss(reconstruction, target, reduction="mean")
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
    return {"total": reconstruction_loss + kl_weight * kl,
            "reconstruction": reconstruction_loss, "kl": kl}
