| Variant | Accuracy | Looping | Cut at limit | Mean tokens | Score | n |
|---|---|---|---|---|---|---|
| base, T=0.6 (blog sampling) | 16.7% | 62.5% | 97.9% | 4027 | 0.185 | 48 |
| base, T=1.0 (Qwen card) | 8.3% | 0.0% | 100.0% | 4096 | 0.333 | 48 |
| published Exp-8 edit (131k weights) | 27.1% | 18.8% | 89.6% | 3845 | 0.383 | 48 |
| published delta x 1.6 | 20.8% | 6.2% | 89.6% | 3878 | 0.396 | 48 |
| published delta x 2.4 | 20.8% | 4.2% | 79.2% | 3831 | 0.433 | 48 |
| published delta x 3.2 | 6.2% | 14.6% | 83.3% | 3858 | 0.331 | 48 |
| optimizer-found edit | 16.7% | 12.5% | 81.2% | 3796 | 0.385 | 48 |
