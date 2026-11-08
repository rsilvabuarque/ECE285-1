"""Dimensionless KL weights with an explicit optimizer-update clock."""
import math


def beta_at_update(*, mode, final, update, epoch, warmup_epochs, reference_updates_per_epoch,cycle_updates=2000):
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
    elif mode == "cyclic_updates":
        if cycle_updates < warmup_epochs*reference_updates_per_epoch:
            raise ValueError("Each cycle must contain the entire reference warmup exposure")
        exposure_epoch = (update % cycle_updates) // reference_updates_per_epoch + 1
    else:
        raise ValueError("Unsupported beta schedule")
    return float(final * min(1., exposure_epoch / warmup_epochs))


def schedule_metadata(mode, final, warmup_epochs, reference_updates_per_epoch,cycle_updates=2000):
    beta_at_update(mode=mode, final=final, update=0, epoch=1,
        warmup_epochs=warmup_epochs, reference_updates_per_epoch=reference_updates_per_epoch,cycle_updates=cycle_updates)
    return dict(mode=mode, final_beta=float(final), warmup_epochs=int(warmup_epochs),
        reference_updates_per_epoch=int(reference_updates_per_epoch), update_index="zero-based before optimizer step",
        nominal_warmup_updates=int(warmup_epochs*reference_updates_per_epoch),
        full_weight_start_update_one_based=int((warmup_epochs-1)*reference_updates_per_epoch+1) if mode in {"reference_updates","cyclic_updates"} else None,
        formula=("beta_final*min(1,(floor((u%cycle_updates)/reference_updates_per_epoch)+1)/warmup_epochs)" if mode=="cyclic_updates" else
            "beta_final*min(1,(floor(u/reference_updates_per_epoch)+1)/warmup_epochs)" if mode=="reference_updates" else "beta_final*min(1,epoch/warmup_epochs)"),
        cycle_updates=int(cycle_updates) if mode=="cyclic_updates" else None,
        cyclic_scope="Repeat the identical reference-block warmup; first cycle matches monotonic baseline, reset is beta_final/warmup_epochs rather than literal zero" if mode=="cyclic_updates" else None,
        methodological_reference="Fu et al.2019 sec3.2, DOI10.18653/v1/N19-1021; discrete adaptation, not their default M4/R0.5" if mode=="cyclic_updates" else None,
        units="Dimensionless beta multiplying a per-row latent KL sum in nats")
