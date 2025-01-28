# PyWPH: Wavelet Phase Harmonics in Python

![PyPI](https://img.shields.io/pypi/v/pywph)
![Python Versions](https://img.shields.io/pypi/pyversions/pywph)
![License](https://img.shields.io/github/license/bregaldo/pywph)

**PyWPH** is a Python package for computing and handling **Wavelet Phase Harmonic (WPH) statistics**.
These statistics can be derived from both real and complex-valued 2D data (e.g., images). Calculations are GPU-accelerated using **PyTorch** (`torch>=1.9.0`). Refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for setting up PyTorch.

## Features

- GPU-accelerated computations with support for low-memory GPUs through efficient chunk-based processing.
- Support for real and complex-valued 2D data.
- Cross-WPH statistics for cross-statistical analysis.
- Ready-to-use examples for syntheses (including multi-channel synthese in external repository) and statistical denoising

## Installation

Install PyWPH via [PyPI](https://pypi.org/project/pywph/):
```bash
pip install pywph
```

Alternatively, install form source:
```bash
git clone https://github.com/bregaldo/pywph.git
cd pywph
pip install .
```
To uninstall:
```bash
pip uninstall pywph
```

## Documentation and Examples

Explore the following resources to get started:
* 📖 [Tutorial](examples/tutorial.ipynb): A step-by-step introduction to PyWPH.
* 📂 [Examples folder](examples/): Basic examples for computing WPH coefficients and advanced applications such as synthesis and statistical denoising.
* 🖼️ Multi-channel synthesis examples are available in [this repository](https://github.com/bregaldo/dust_genmodels).

For a detailed presentation of the WPH statistics implemented in this package, refer to the paper: [arXiv:2208.03538](https://arxiv.org/abs/2208.03538).

## Citing PyWPH

If you use PyWPH in your research, please cite the following paper:
* Regaldo-Saint Blancard, B., Allys, E., Boulanger, F., Levrier, F., & Jeffrey, N. "A new approach for the statistical denoising of Planck interstellar dust polarization data", [*Astronomy \& Astrophysics 649, L18*](https://doi.org/10.1051/0004-6361/202140503) (2021). ArXiv: [2102.03160](https://arxiv.org/abs/2102.03160)

```
@article{regaldo2021,
       author = {{Regaldo-Saint Blancard}, Bruno and {Allys}, Erwan and {Boulanger}, Fran{\c{c}}ois and {Levrier}, Fran{\c{c}}ois and {Jeffrey}, Niall},
        title = "{A new approach for the statistical denoising of Planck interstellar dust polarization data}",
      journal = {Astronomy \& Astrophysics},
         year = 2021,
        month = may,
       volume = {649},
          eid = {L18},
        pages = {L18},
          doi = {10.1051/0004-6361/202140503},
archivePrefix = {arXiv},
       eprint = {2102.03160},
 primaryClass = {astro-ph.CO},
}
```


## Related References

* Mallat, S., Zhang, S., & Rochette, G. "Phase harmonic correlations and convolutional neural networks", [*Information and Inference: A Journal of the IMA, 9(3), 721–747.*](https://doi.org/10.1093/imaiai/iaz019) (2020). ArXiv: [1810.12136](https://arxiv.org/abs/1810.12136).
* Allys, E., Marchand, T., Cardoso, J.-F., Villaescusa-Navarro, F., Ho, S., & Mallat, S. "New Interpretable Statistics for Large Scale Structure Analysis and Generation", [*Physical Review D, 102(10), 103506.*](https://doi.org/10.1103/PhysRevD.102.103506) (2020). ArXiv: [2006.06298](http://arxiv.org/abs/2006.06298).
* Zhang, S., & Mallat, S. "Maximum Entropy Models from Phase Harmonic Covariances", [*Applied and Computational Harmonic Analysis, 53, 199–230*](https://doi.org/10.1016/j.acha.2021.01.003) (2021). ArXiv: [1911.10017](https://arxiv.org/abs/1911.10017).
* Régaldo-Saint Blancard, B., Allys, E., Auclair, C., Boulanger, F., Eickenberg, M., Levrier, F., Vacher, L. & Zhang, S. "Generative Models of Multi-channel Data from a Single Example - Application to Dust Emission", [*The Astrophysical Journal, 943, 9*](https://doi.org/10.3847/1538-4357/aca538) (2023) ArXiv: [2208.03538](https://arxiv.org/abs/2208.03538). [Code](https://github.com/bregaldo/dust_genmodels).

This package took inspiration from [https://github.com/Ttantto/wph_quijote](https://github.com/Ttantto/wph_quijote).

## Changelog

### v1.1

* New default discretization grid for the shift vector $\tau$.
* New set of scaling moments $L$ (which replaced the old ones).
* Version used in [arXiv:2208.03538](https://arxiv.org/abs/2208.03538).

### v1.0

* Added cross-WPH statistics.
* Smarter way to evaluate moments at different $\tau$.
* Improved computation for non-periodic boundary conditions data.

### v0.9

* Initial release, corresponding to [arXiv:2102.03160](https://arxiv.org/abs/2102.03160).
