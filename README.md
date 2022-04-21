# PyWPH : Wavelet Phase Harmonics in Python

PyWPH is a Python package designed for the computation and handling of the Wavelet Phase Harmonics (WPH) statistics.
These statistics can be computed from real or complex images (2D data). Calculations are GPU accelerated using PyTorch 1.8/CUDA.

Install PyWPH and check out the examples/ folder. You will find elementary examples to compute WPH coefficients from an image,
as well as more convoluted synthesis or denoising scripts.

This code is a rework and extension of [https://github.com/Ttantto/wph_quijote](https://github.com/Ttantto/wph_quijote).

If you use this package, please cite the following paper:
* Regaldo-Saint Blancard, B., Allys, E., Boulanger, F., Levrier, F., & Jeffrey, N. (2021). A new approach for the statistical denoising of Planck interstellar dust polarization data. [arXiv:2102.03160](https://arxiv.org/abs/2102.03160)

Related references:
* Mallat, S., Zhang, S., & Rochette, G. (2020). Phase harmonic correlations and convolutional neural networks. Information and Inference: A Journal of the IMA, 9(3), 721–747. https://doi.org/10.1093/imaiai/iaz019 [arXiv:1810.12136](https://arxiv.org/abs/1810.12136)
* Allys, E., Marchand, T., Cardoso, J.-F., Villaescusa-Navarro, F., Ho, S., & Mallat, S. (2020). New Interpretable Statistics for Large Scale Structure Analysis and Generation. Physical Review D, 102(10), 103506. [arXiv:2006.06298](http://arxiv.org/abs/2006.06298)
* Zhang, S., & Mallat, S. (2021). Maximum Entropy Models from Phase Harmonic Covariances. Applied and Computational Harmonic Analysis, 53, 199–230. https://doi.org/10.1016/j.acha.2021.01.003 [arXiv:1911.10017](https://arxiv.org/abs/1911.10017)

## Install/Uninstall

### Standard installation (from the Python Package Index)

```
pip install pywph
```

### Install from source

Clone the repository and type from the main directory:

```
pip install -r requirements.txt
pip install .
```

### Uninstall

```
pip uninstall pywph
```

## Changelog
