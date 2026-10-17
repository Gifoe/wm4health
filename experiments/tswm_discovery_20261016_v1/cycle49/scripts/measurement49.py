"""Classical marked-Poisson transforms and unbiased finite-ensemble scoring.

Known Gaussian observation covariance is diagonal in these utilities.
Rates are expected counts per bin, not rates per unit time.
"""
import torch


def conditional_features(mean, expected_counts, count_probes, mark_probes, noise_std):
    """mean[...,time,dim], counts[...,time], probes[probe,time,(dim)]."""
    angle = torch.einsum('...td,ftd->...ft', mean, mark_probes)
    variance = (mark_probes * noise_std).square().sum(-1)
    mark_transform = torch.exp(torch.complex(-0.5 * variance, angle + count_probes))
    exponent = (expected_counts.unsqueeze(-2) * (mark_transform - 1)).sum(-1)
    return exponent.exp()


def observed_features(counts, mark_sums, count_probes, mark_probes):
    angle = (torch.einsum('...t,ft->...f', counts, count_probes)
             + torch.einsum('...td,ftd->...f', mark_sums, mark_probes))
    return torch.exp(torch.complex(torch.zeros_like(angle), angle))


def unbiased_score(features, observed):
    """Independent future samples on axis -2; return one score per probe.

    The diagonal subtraction is necessary even for non-unit-modulus
    observation-marginalized features.
    """
    n = features.shape[-2]
    if n < 2:
        raise ValueError('At least two independent latent futures are required')
    total = features.sum(-2)
    pair = (total.abs().square() - features.abs().square().sum(-2)) / (n * (n - 1))
    return pair - 2 * (features.mean(-2) * observed.conj()).real
