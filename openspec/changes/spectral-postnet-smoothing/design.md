## Context

`PostNet` (`eeggan/nn_architecture/models.py:24`) is a Tacotron2-style time-domain residual conv stack, added by `openspec/changes/archive/2026-08-29-generator-postnet-smoothing` to suppress patch-boundary seam artifacts in `TTSGenerator`'s output. It works visually but the ablation grid (`comparison_plots/metrics_*.csv`) shows it systematically worsens `jsd_mean`, `mmd2`, `erp_amp_err` and `erp_lat_err_ms` on every config it's paired with. A time-domain conv stack has no way to know that seam artifacts and the genuine P300 peak transition are different things — both are fast local changes in the raw signal — so it smooths both indiscriminately.

The metrics themselves already give the fix its shape: `compare_samples.py`'s `plot_psd_overlay` computes `jsd_mean` only within `[0, BP_HZ]` Hz (`BP_HZ=24`, matching `moabb.paradigms.P300().filters == [[1, 24]]`, the band real BNCI2014_009 trials are actually filtered to). The visual artifact this line of work is chasing lives *above* that band by definition — real data has nothing there to imitate. A module that structurally cannot touch anything below 24 Hz, and can only attenuate above it, targets exactly the region where the artifact lives and nowhere else.

## Goals / Non-Goals

**Goals:**
- Remove patch-boundary seam energy (the out-of-band content PostNet already proved it can suppress) without perturbing anything within the real bandpass — i.e. without the collateral damage to `erp_amp_err`/`erp_lat_err_ms`/`jsd_mean` that the time-domain PostNet causes.
- End-to-end trainable with the existing adversarial + feature-matching losses, no separate pretraining or auxiliary network — same constraint `PostNet` already satisfies.
- Coexist with `PostNet` as an independent, separately-flagged option (`use_spectral_postnet` alongside `use_postnet`), so the ablation can compare them head-to-head on the same generator/discriminator setup.
- Backward compatible: default off, old checkpoints unaffected.

**Non-Goals:**
- Not removing or replacing `PostNet` — it stays as-is; both existed for the ablation grid before this change, this just adds a third option (neither / PostNet / SpectralPostNet, with combinations technically possible but not the primary comparison).
- Not making the below-cutoff bins trainable, even as a stretch goal — pinning them is the entire mechanism that protects P300 morphology. If this turns out to be too restrictive (seam energy leaking below 24 Hz), that's a follow-up change, not a parameter to sweep here.
- Not adding a smooth/soft transition band at the cutoff in this first version — a hard bin split (below = pinned, at-or-above = learnable) is simpler and matches `compare_samples.py`'s own hard `fmax=BP_HZ` cutoff in `spectral_jsd`. Revisit only if a visible discontinuity artifact appears at 24 Hz in the PSD plots.

## Decisions

**1. Frequency-domain gain vector via `torch.fft.rfft`/`irfft`, not a learned filter kernel or spectral-norm layer.**
`rfft` over the time axis gives one complex coefficient per frequency bin; multiplying by a real learnable gain per bin (and taking `irfft`) is the simplest differentiable operation that maps directly onto "attenuate these frequencies, leave those alone" — which is exactly the requirement. A DSP-style FIR/IIR filter would need its coefficients cast into a comparably differentiable form for no benefit over the direct bin-gain approach at this signal length (per-trial time axis is short — `seq_len`, not spanning a large FFT size where FIR would matter). `torch.fft.rfft`/`irfft` support autograd directly; no custom backward needed.

**2. Bins below the cutoff are excluded from the learnable parameter entirely (not just initialized to 1 and left free).**
The design doc for the original PostNet change already tried "additive residual, hope it stays close to identity" — that's exactly the mechanism that let PostNet drift into damaging in-band content, because nothing structurally prevented it. Here, the gain vector is split at construction time into a fixed buffer (`ones`, bins `< cutoff_bin`, `requires_grad=False`) and a learnable `nn.Parameter` (bins `>= cutoff_bin`, initialized to `1.0`). The full gain applied is `torch.cat([fixed, learnable])` — gradients literally cannot reach the protected bins. This is a stronger guarantee than "trained not to," which is what the risk section of the original PostNet design leaned on and what the ablation data shows failed in practice.

**3. Cutoff bin computed from `seq_len` and a `sample_rate` constructor arg (default `256`, matching `compare_samples.py`'s `FS`), not hardcoded to a fixed bin index.**
`TTSGenerator` is used across different `seq_len`/`patch_size` configs in this codebase (see `CLAUDE.md` §7, patch divisibility constraint); a fixed bin index would silently target the wrong frequency if `seq_len` changes. `cutoff_bin = ceil(BP_HZ * seq_len / sample_rate)` (rfft bin `k` corresponds to `k * sample_rate / seq_len` Hz) keeps the 24 Hz semantic meaning stable across configs. `BP_HZ` itself stays a `SpectralPostNet` constructor default (`24.0`), overridable, rather than importing `compare_samples.py`'s constant (that module is an evaluation script, not a package dependency `models.py` should import from).

**4. Lives in `eeggan/nn_architecture/models.py` next to `PostNet`, same wiring pattern (`TTSGenerator.__init__`/`forward`).**
Consistent with Decision 2 of the original PostNet design (keeps vendored `tts_gan_components.py` untouched) and with how this codebase already threads `use_postnet` end-to-end (`system_inputs.py` → `gan_training_main.py opt` → `initialize_gan.py` → constructor → `trainer.py configuration` → `generate_samples_main.py`). `use_spectral_postnet` follows the identical path as a sibling flag.

**5. `use_postnet` and `use_spectral_postnet` are independent booleans, both defaulting `False`; `TTSGenerator.forward` applies whichever are enabled, in sequence (`PostNet` first, then `SpectralPostNet`, both residual).**
Simpler than a single enum (`postnet_mode='none'|'time'|'spectral'`) for the same reason the codebase already prefers flat boolean flags (`use_multiscale_dwt_discriminator`, `use_stacking`, `use_postnet`) over combined mode switches — consistent with existing convention, and doesn't foreclose testing both stacked together later without a schema change. The ablation will train configs with one or the other, not typically both, but nothing enforces that.

## Risks / Trade-offs

- **[Risk]** A checkpoint trained with `use_spectral_postnet=True` has extra `state_dict` keys (`spectral_postnet.*`, including the non-persistent fixed buffer); loading it into a generator built without the flag raises a `state_dict` mismatch — same failure mode Decision 4 of the original PostNet design already identified and mitigated for `use_postnet`. → **Mitigation**: same fix, applied here too — `generate_samples_main.py` reads `use_spectral_postnet` from the checkpoint's saved `configuration`, not from CLI/default, before reconstructing the generator.
- **[Risk]** Hard bin cutoff could leave a small amount of seam energy just below 24 Hz untouched (real spectral leakage doesn't respect bin boundaries cleanly, especially at short `seq_len` where each bin spans several Hz). → **Mitigation**: accepted for this first version (Non-Goal above); if PSD plots after training still show a visible bump just under 24 Hz, a soft/smoothed transition band is a small, isolated follow-up change to `SpectralPostNet.__init__`'s mask construction, not a redesign.
- **[Risk]** `rfft`/`irfft` on very short `seq_len` (small ablation configs) gives coarse bin resolution — `cutoff_bin` could round to 0 or to `seq_len//2` at extreme `patch_size`/`seq_len` combinations, degenerating to "everything learnable" or "nothing learnable." → **Mitigation**: `SpectralPostNet.__init__` should assert `0 < cutoff_bin < n_bins` and raise a clear error otherwise, so a misconfigured combination fails fast at construction instead of silently no-op'ing or silently unprotecting P300 morphology.
- **[Trade-off]** Doesn't address the seam mechanism itself any more than `PostNet` does — still corrective (post-hoc gain on the generator's own raw output), not preventive (overlap-add patch reconstruction). Same trade-off the original PostNet design already accepted (Decision 1 there); out of scope here too.

## Migration Plan

Purely additive: default `False` reproduces current architecture exactly (no `spectral_postnet` submodule instantiated, `forward` path unchanged). Existing checkpoints have no `use_spectral_postnet` key in saved `configuration`; both the writing side (`trainer.py`/`gan_training_main.py`) and reading side (`generate_samples_main.py`/`get_gan_config.py`) default missing `use_spectral_postnet` to `False` via `.get('use_spectral_postnet', False)`. No rollback beyond training with the flag off (the default).

## Open Questions

- Should `SpectralPostNet` eventually be combined with `PostNet` in the same generator (both flags on) as its own ablation arm, or is that out of scope until the head-to-head result is in? Leaning: skip for now, revisit only if `SpectralPostNet` alone doesn't fully close the visual gap.
