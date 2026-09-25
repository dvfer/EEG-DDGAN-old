## Why

The existing `PostNet` (`openspec/specs/generator-postnet-smoothing/`) suppresses `TTSGenerator`'s patch-boundary seam artifacts and makes synthetic EEG *look* cleaner, but ablation data shows it degrades every quantitative fidelity metric it should be neutral to: comparing matched configs with/without it (e.g. `fm_lambda_50` vs `fm_lambda_50_postnet`), `jsd_mean`, `mmd2`, `erp_amp_err` and `erp_lat_err_ms` all get worse. The reason is structural, not a tuning problem: `PostNet` is a time-domain residual conv stack, and a time-domain filter cannot distinguish "spurious seam energy" from "genuine P300 peak transition" — both are fast, localized changes in the same signal. It smooths both. This also means it's fighting a metric it can't actually fail on: `jsd_mean` is already computed only within `[0, BP_HZ]` Hz (`compare_samples.py:184`, `BP_HZ=24`, matching MOABB's `P300()` bandpass), i.e. exactly the band real data was filtered to — the visual artifact PostNet was built to remove lives *above* that band, where none of the current metrics even look.

## What Changes

- Add `SpectralPostNet`, a frequency-domain alternative to `PostNet`: applies `torch.fft.rfft` over the time axis, multiplies by a learnable per-frequency-bin real gain, applies `torch.fft.irfft`, residual-added to the generator's raw output exactly like `PostNet` is. Bins below the real bandpass cutoff (24 Hz) are pinned to gain `1.0` and excluded from the learnable parameters, so P300 morphology (which lives below that cutoff) cannot be attenuated by construction — only the out-of-band seam energy is ever eligible to be learned away.
- Gate it behind a new CLI flag `use_spectral_postnet` (default `False`), wired the same way `use_postnet` already is: `system_inputs.py` schema → `gan_training_main.py` `opt` dict → `initialize_gan.py` → `TTSGenerator` constructor.
- `use_postnet` and `use_spectral_postnet` are independent flags — either, both, or neither may be set on a given generator. This ablation will primarily compare them head-to-head (not combined), but nothing prevents combining them later.
- Expose `GAN_USE_SPECTRAL_POSTNET` in `moabb_pipeline.py`'s GAN config block, threaded through `train_gan()`.
- Record `use_spectral_postnet` in the trainer's saved `configuration` dict (same place `use_postnet`/`lambda_fm` live) so a checkpoint self-describes its architecture for `generate_samples_main.py`/`get_gan_config.py`.

## Capabilities

### New Capabilities
- `spectral-postnet-smoothing`: optional end-to-end-trained frequency-domain gain applied to the generator's output spectrum, band-locked so only above-cutoff (seam-artifact) frequencies are ever attenuated and below-cutoff (P300 morphology) content is structurally protected.

### Modified Capabilities
(none — additive only, `generator-postnet-smoothing` is unchanged and untouched by this change)

## Impact

- `eeggan/nn_architecture/models.py` — new `SpectralPostNet` module; `TTSGenerator` gains a second optional smoothing stage, independent of the existing `postnet` attribute.
- `eeggan/helpers/initialize_gan.py` — thread `use_spectral_postnet` into `TTSGenerator` construction.
- `eeggan/helpers/system_inputs.py` — new `use_spectral_postnet` CLI flag.
- `eeggan/gan_training_main.py` — `opt['use_spectral_postnet']`, recorded in checkpoint `configuration`.
- `eeggan/helpers/trainer.py` — `configuration` dict gains `use_spectral_postnet` (read-through only).
- `eeggan/generate_samples_main.py` — must read `use_spectral_postnet` from the checkpoint's saved configuration when rebuilding the generator for inference.
- `moabb_pipeline.py` — new `GAN_USE_SPECTRAL_POSTNET` config var, threaded through `train_gan()`.
- Backward compatible: default `False` reproduces current architecture byte-for-byte; existing checkpoints (saved without `use_spectral_postnet`) load as `False` via `.get(..., False)`.
