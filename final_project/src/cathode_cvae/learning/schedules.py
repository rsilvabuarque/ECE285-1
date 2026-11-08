"""Dimensionless KL weights with an explicit optimizer-update clock."""
import math


def beta_at_update(*, mode, final, update, epoch, warmup_epochs, reference_updates_per_epoch):
    """Return the weight BEFORE zero-based optimizer update ``update``.

    reference_updates reproduces the old piecewise-constant epoch exposure on
    the reference dataset even when another dataset has shorter natural epochs.
    It intentionally starts at final/warmup_epochs, matching the legacy writer.
    """
    if not math.isfinite(final) or final < 0 or update < 0 or epoch < 1 or warmup_epochs < 1 or reference_updates_per_epoch < 1:
        raise ValueError("Invalid beta schedule parameters")
    if mode == "legacy_epoch":
        exposure_epoch = epoch
    elif mode == "reference_updates":
        exposure_epoch = update // reference_updates_per_epoch + 1
    else:
        raise ValueError("Unsupported beta schedule")
    return float(final * min(1., exposure_epoch / warmup_epochs))


def schedule_metadata(mode, final, warmup_epochs, reference_updates_per_epoch):
    beta_at_update(mode=mode, final=final, update=0, epoch=1,
        warmup_epochs=warmup_epochs, reference_updates_per_epoch=reference_updates_per_epoch)
    return dict(mode=mode, final_beta=float(final), warmup_epochs=int(warmup_epochs),
        reference_updates_per_epoch=int(reference_updates_per_epoch), update_index="zero-based before optimizer step",
        nominal_warmup_updates=int(warmup_epochs*reference_updates_per_epoch),
        full_weight_start_update_one_based=int((warmup_epochs-1)*reference_updates_per_epoch+1) if mode=="reference_updates" else None,
        formula="beta_final*min(1,(floor(u/reference_updates_per_epoch)+1)/warmup_epochs)" if mode=="reference_updates" else "beta_final*min(1,epoch/warmup_epochs)",
        units="Dimensionless beta multiplying a per-row latent KL sum in nats")
