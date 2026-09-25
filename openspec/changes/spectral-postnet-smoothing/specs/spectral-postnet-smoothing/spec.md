## ADDED Requirements

### Requirement: Optional frequency-domain gain on generator output
`TTSGenerator` SHALL support an optional `SpectralPostNet` stage applied to its raw output, controlled by a boolean `use_spectral_postnet` flag passed at construction, independent of the existing `use_postnet` flag. When disabled, the generator's forward pass SHALL be byte-for-byte identical to the architecture without this change.

#### Scenario: SpectralPostNet disabled (default)
- **WHEN** `TTSGenerator` is constructed with `use_spectral_postnet=False` (or the argument omitted)
- **THEN** no `SpectralPostNet` submodule is instantiated and `forward()`'s output is unaffected by this change

#### Scenario: SpectralPostNet enabled
- **WHEN** `TTSGenerator` is constructed with `use_spectral_postnet=True`
- **THEN** a `SpectralPostNet` submodule is instantiated and its output is added (residual) to the generator's raw output before `forward()` returns, preserving the output tensor's shape `(batch, seq_len, channels)`

#### Scenario: Both smoothing stages enabled together
- **WHEN** `TTSGenerator` is constructed with both `use_postnet=True` and `use_spectral_postnet=True`
- **THEN** both residual stages are applied (`PostNet` then `SpectralPostNet`), and neither flag's behavior depends on the other's value

### Requirement: Below-cutoff frequencies structurally protected from attenuation
`SpectralPostNet` SHALL apply a real gain per `rfft` frequency bin to the generator's output spectrum. Bins corresponding to frequencies below a configurable cutoff (default `24.0` Hz) SHALL be fixed at gain `1.0` and excluded from the module's learnable parameters, so no gradient update can ever attenuate them. Only bins at or above the cutoff SHALL be learnable, initialized to gain `1.0`.

#### Scenario: Gradient flow respects the cutoff
- **WHEN** `SpectralPostNet` is trained via backpropagation through the generator loss
- **THEN** the fixed below-cutoff gains never change value across training steps, and only the learnable at-or-above-cutoff gains receive nonzero gradients

#### Scenario: Identity at initialization
- **WHEN** a freshly constructed `SpectralPostNet` (untrained) is applied to any input
- **THEN** its output is numerically equivalent to applying an all-ones gain vector (i.e. `irfft(rfft(x)) ≈ x`, a no-op up to floating-point tolerance)

#### Scenario: Cutoff mapped from sample rate and sequence length
- **WHEN** `SpectralPostNet` is constructed with a given `seq_len` and `sample_rate`
- **THEN** the cutoff bin index is computed from `cutoff_hz`, `sample_rate`, and `seq_len` (rfft bin `k` corresponds to `k * sample_rate / seq_len` Hz), not hardcoded to a fixed bin index

#### Scenario: Degenerate configuration fails fast
- **WHEN** `SpectralPostNet` is constructed with a `seq_len`/`sample_rate`/`cutoff_hz` combination whose computed cutoff bin is `0` or `>=` the number of rfft bins
- **THEN** construction raises a clear error instead of silently producing an all-learnable or all-fixed gain vector

### Requirement: End-to-end training, no separate coupling
`SpectralPostNet` SHALL be trained jointly with the rest of the generator using the existing adversarial and feature-matching losses. It SHALL NOT require a separate pretraining stage, a frozen auxiliary network, or an additional encode/decode step to feed any discriminator.

#### Scenario: Training with SpectralPostNet enabled
- **WHEN** a GAN is trained with `use_spectral_postnet=True`
- **THEN** gradients from the generator loss (adversarial + feature-matching, including through `MultiscaleDWTDiscriminator` when active) flow into `SpectralPostNet`'s learnable gains via the same backward pass as the rest of the generator, with no extra forward/backward pass through any other network

### Requirement: CLI-parametrizable
`use_spectral_postnet` SHALL be exposed as a CLI argument for `gan_training` following the existing `key=value` / bare-flag parsing convention, defaulting to `False`.

#### Scenario: Default training run
- **WHEN** `eeggan gan_training ...` is invoked without `use_spectral_postnet`
- **THEN** the GAN trains with `use_spectral_postnet=False`

#### Scenario: Opt-in training run
- **WHEN** `eeggan gan_training ... use_spectral_postnet` (or `use_spectral_postnet=True`) is invoked
- **THEN** the GAN trains with the SpectralPostNet stage enabled

### Requirement: moabb_pipeline.py config exposure
`moabb_pipeline.py` SHALL expose a `GAN_USE_SPECTRAL_POSTNET` config variable in its GAN configuration block, threaded through `train_gan()` into the `eeggan gan_training` CLI call.

#### Scenario: Pipeline run with SpectralPostNet toggled
- **WHEN** `GAN_USE_SPECTRAL_POSTNET = True` is set in `moabb_pipeline.py` and the pipeline is run
- **THEN** every subject's `eeggan gan_training` invocation includes `use_spectral_postnet` (or `use_spectral_postnet=True`)

### Requirement: Checkpoint self-describes SpectralPostNet presence
A trained checkpoint's saved configuration SHALL record whether its generator was built with `use_spectral_postnet=True`, so that reconstructing the generator architecture at inference time does not depend on the caller passing the correct flag.

#### Scenario: Loading a SpectralPostNet checkpoint for generation
- **WHEN** `generate_samples_main.py` loads a checkpoint whose saved configuration has `use_spectral_postnet=True`
- **THEN** it reconstructs `TTSGenerator` with `use_spectral_postnet=True` before loading the saved `state_dict`, without requiring the user to pass `use_spectral_postnet` on the `generate_samples` command line

#### Scenario: Loading a pre-existing checkpoint without the field
- **WHEN** `generate_samples_main.py` loads a checkpoint saved before this change (no `use_spectral_postnet` key in its configuration)
- **THEN** it treats `use_spectral_postnet` as `False` and reconstructs the generator without the stage, matching how that checkpoint was originally trained
