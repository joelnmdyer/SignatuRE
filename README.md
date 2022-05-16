# SignatuRE
Code for AISTATS 2022 paper `Amortised Likelihood-free Inference for Expensive Time-series Simulators with Signatured Ratio Estimation (SignatuRE)`

Structure of project:
- `analysis`: contains notebook for analysing results
- `data`: contains pseudo-observations for `OU`, `MA2`, and `GSE` simulators
- `inference`: implements kernel classifiers
- `models`: simulators for `OU`, `MA2`, and `GSE` models
- `utils`: code for computing performance metrics, defining prior densities, defining embedding networks, and posterior sampling

Main script: `traing_and_sample.py`

## Citation
Please use the following citation:
```
@InProceedings{pmlr-v151-dyer22a,
  title = 	 { Amortised Likelihood-free Inference for Expensive Time-series Simulators with Signatured Ratio Estimation },
  author =       {Dyer, Joel and Cannon, Patrick W. and Schmon, Sebastian M.},
  booktitle = 	 {Proceedings of The 25th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {11131--11144},
  year = 	 {2022},
  editor = 	 {Camps-Valls, Gustau and Ruiz, Francisco J. R. and Valera, Isabel},
  volume = 	 {151},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {28--30 Mar},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v151/dyer22a/dyer22a.pdf},
  url = 	 {https://proceedings.mlr.press/v151/dyer22a.html}
}
```
