# References

Accessed 2026-10-20 unless noted. Research papers are cited with their arXiv/IACR identifiers. Software documentation is cited by URL; behaviour that matters for this project was also verified in the installed source code (versions in `requirements.lock`).

## Software and documentation

- **[F1]** Flower framework documentation. https://flower.ai/docs/framework/
- **[F2]** Flower: "Use Differential Privacy". https://flower.ai/docs/framework/how-to-use-differential-privacy.html
- **[F3]** Flower example `flower-secure-aggregation` (SecAgg+ with `SecAggPlusWorkflow` and `secaggplus_mod`). https://github.com/adap/flower/tree/main/examples/flower-secure-aggregation
- **[F4]** Flower API reference: `flwr.server.workflow.SecAggPlusWorkflow`. https://flower.ai/docs/framework/ref-api/flwr.server.workflow.SecAggPlusWorkflow.html
- **[F5]** Flower: "Use built-in mods". https://flower.ai/docs/framework/how-to-use-built-in-mods.html
- **[O1]** Opacus documentation. https://opacus.ai/
- **[P1]** PyTorch dev-discuss: "PyTorch macOS x86 builds deprecation starting January 2024". https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690
- **[P2]** PyTorch 2.2 release blog ("PyTorch 2.2.x will be the last version that supports macOS x64"). https://pytorch.org/blog/pytorch2-2/
- **[P3]** PyTorch documentation. https://pytorch.org/docs/stable/

## Datasets

- **[D1]** Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B (CC BY 4.0)
- **[D2]** Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X (CC BY 4.0)
- **[D3]** Ogier du Terrail, J., et al. (2022). FLamby: Datasets and Benchmarks for Cross-Silo Federated Learning in Realistic Healthcare Settings. *NeurIPS 2022 Datasets and Benchmarks*. arXiv:2210.04620. Code: https://github.com/owkin/FLamby
- **[D4]** Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. *Proc. SPIE 1905, Biomedical Image Processing and Biomedical Visualization*.
- **[D5]** Notes on zero-coded missing values in Pima Indians Diabetes, e.g. the `PimaIndiansDiabetes2` corrected version documented in R package references (Wahba et al. 1995; Ripley 1996). https://search.r-project.org/CRAN/refmans/spinifex/html/PimaIndiansDiabetes_wide.html
- **[D6]** Detrano, R., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(5), 304–310.

## Research papers

- **[R1]** McMahan, B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS*. arXiv:1602.05629
- **[R2]** Abadi, M., et al. (2016). Deep Learning with Differential Privacy. *ACM CCS*. arXiv:1607.00133
- **[R3]** Mironov, I. (2017). Rényi Differential Privacy. *IEEE CSF*. arXiv:1702.07476
- **[R4]** Mironov, I., Talwar, K., & Zhang, L. (2019). Rényi Differential Privacy of the Sampled Gaussian Mechanism. arXiv:1908.10530
- **[R5]** Bonawitz, K., et al. (2017). Practical Secure Aggregation for Privacy-Preserving Machine Learning. *ACM CCS*. IACR ePrint 2017/281
- **[R6]** Bell, J. H., Bonawitz, K. A., Gascón, A., Lepoint, T., & Raykova, M. (2020). Secure Single-Server Aggregation with (Poly)Logarithmic Overhead. *ACM CCS*. IACR ePrint 2020/704
- **[R7]** Zhao, Y., et al. (2018). Federated Learning with Non-IID Data. arXiv:1806.00582
- **[R8]** Li, T., et al. (2020). Federated Optimization in Heterogeneous Networks (FedProx). *MLSys*. arXiv:1812.06127
- **[R9]** Hsu, T.-M. H., Qi, H., & Brown, M. (2019). Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification. arXiv:1909.06335
- **[R10]** Rieke, N., et al. (2020). The future of digital health with federated learning. *npj Digital Medicine*, 3, 119. arXiv:2003.08119
- **[R11]** Kaissis, G. A., Makowski, M. R., Rückert, D., & Braren, R. F. (2020). Secure, privacy-preserving and federated machine learning in medical imaging. *Nature Machine Intelligence*, 2, 305–311.
- **[R12]** Yousefpour, A., et al. (2021). Opacus: User-Friendly Differential Privacy Library in PyTorch. arXiv:2109.12298
- **[R13]** McMahan, H. B., Ramage, D., Talwar, K., & Zhang, L. (2018). Learning Differentially Private Recurrent Language Models (client-level DP). *ICLR*. arXiv:1710.06963
- **[R14]** Kairouz, P., et al. (2021). Advances and Open Problems in Federated Learning. *Foundations and Trends in ML*. arXiv:1912.04977
- **[R15]** Zhu, L., Liu, Z., & Han, S. (2019). Deep Leakage from Gradients. *NeurIPS*. arXiv:1906.08935
- **[R16]** Geiping, J., Bauermeister, H., Dröge, H., & Moeller, M. (2020). Inverting Gradients: How easy is it to break privacy in federated learning? *NeurIPS*. arXiv:2003.14053
- **[R17]** Beutel, D. J., et al. (2020). Flower: A Friendly Federated Learning Research Framework. arXiv:2007.14390
- **[R18]** Dwork, C., & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy. *Foundations and Trends in Theoretical Computer Science*, 9(3–4).
- **[R19]** Sheller, M. J., et al. (2020). Federated learning in medicine: facilitating multi-institutional collaborations without sharing patient data. *Scientific Reports*, 10, 12598.
