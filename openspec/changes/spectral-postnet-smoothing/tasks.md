## 1. `SpectralPostNet` module

- [ ] 1.1 In `eeggan/nn_architecture/models.py`, add `SpectralPostNet(nn.Module)` next to `PostNet` (~line 24): constructor takes `channels`, `seq_len`, `sample_rate=256`, `cutoff_hz=24.0`; computes `n_bins = seq_len // 2 + 1` and `cutoff_bin = ceil(cutoff_hz * seq_len / sample_rate)`; raises `ValueError` if `not (0 < cutoff_bin < n_bins)` (degenerate-config guard from design.md Risks).
- [ ] 1.2 Register a non-persistent fixed buffer `ones(cutoff_bin)` for below-cutoff gains and an `nn.Parameter(ones(n_bins - cutoff_bin))` for at-or-above-cutoff gains; `forward(x)` builds `gain = torch.cat([fixed, learnable])`, applies `x_freq = torch.fft.rfft(x, dim=1)`, multiplies by `gain` broadcast over the channel dim, `torch.fft.irfft(x_freq * gain, n=seq_len, dim=1)`; input/output shape `(B, L, C)` matching `PostNet.forward`'s convention.
- [ ] 1.3 Wire into `TTSGenerator.__init__` (~line 53): add `use_spectral_postnet=False` param, `self.spectral_postnet = SpectralPostNet(channels, seq_len) if use_spectral_postnet else None`.
- [ ] 1.4 Wire into `TTSGenerator.forward` (~line 60): after the existing `if self.postnet is not None` block, add `if self.spectral_postnet is not None: output = output + self.spectral_postnet(output)` — independent of the `postnet` branch, both may fire.

## 2. CLI / config plumbing (mirrors existing `use_postnet` wiring)

- [ ] 2.1 `eeggan/helpers/system_inputs.py:284` area — add `'use_spectral_postnet': [bool, 'Use frequency-domain SpectralPostNet gain on generator output (band-locked below 24Hz)', False]` next to the `use_postnet` entry.
- [ ] 2.2 `eeggan/helpers/initialize_gan.py:5,48` — add `use_spectral_postnet=False` to the generator-building lambda/kwargs signature and pass `use_spectral_postnet=kwargs.get('use_spectral_postnet', False)` through to `TTSGenerator`.
- [ ] 2.3 `eeggan/gan_training_main.py:109` — add `'use_spectral_postnet': default_args.get('use_spectral_postnet', False)` to the `opt` dict next to `use_postnet`.
- [ ] 2.4 `eeggan/helpers/trainer.py:124` — add `'use_spectral_postnet': opt.get('use_spectral_postnet', False)` to the saved `configuration` dict.
- [ ] 2.5 `eeggan/generate_samples_main.py:131` — add `use_spectral_postnet=state_dict['configuration'].get('use_spectral_postnet', False)` to the generator reconstruction call.
- [ ] 2.6 `moabb_pipeline.py` — add `GAN_USE_SPECTRAL_POSTNET = False` to the GAN config block; add `use_spectral_postnet=False` param to `train_gan()` (~line 213), append `'use_spectral_postnet'` to `args` when set (~line 235-236 pattern), pass `use_spectral_postnet=GAN_USE_SPECTRAL_POSTNET` at the `train_gan()` call site (~line 324).

## 3. Self-check

- [ ] 3.1 Add `_demo_spectral_postnet_self_check()` next to `_demo_postnet_self_check()` (~line 317), asserting: (a) `use_spectral_postnet=False` is a byte-for-byte no-op vs. default construction (mirrors the existing PostNet check); (b) a freshly constructed `SpectralPostNet` alone is numerically ≈ identity (`torch.allclose` within float tolerance) on a random input; (c) after one manual gradient step, only the learnable (>=cutoff) gains have moved from `1.0` and the fixed (<cutoff) gains are still exactly `1.0`; (d) gradients reach `spectral_postnet.parameters()` when `use_spectral_postnet=True`; (e) a degenerate `seq_len`/`cutoff_hz` combination raises `ValueError` at construction. Call it from `__main__` alongside the existing check.
- [ ] 3.2 Run `python eeggan/nn_architecture/models.py` and confirm both self-checks print OK.

## 4. Ablation wiring (not training itself — that's a separate manual step)

- [ ] 4.1 Confirm `eeggan gan_training data=<small_csv> ... use_spectral_postnet` runs end-to-end on a small existing subject CSV (smoke test, few epochs) without shape/key errors.
- [ ] 4.2 Confirm `eeggan generate_samples model=<that_checkpoint>.pt ...` reconstructs the generator and generates without requiring `use_spectral_postnet` on the command line.
- [ ] 4.3 Confirm `compare_samples.py`/`eval_external_config.py`-style evaluation runs against the smoke-test checkpoint and produces `jsd_mean`/`mmd2`/`erp_amp_err`/`erp_lat_err_ms` without errors (does not need to beat any baseline yet — just prove the plumbing).
