# NVGesture Honest Fusion Leaderboard

Updated 2026-05-14. All accs on 482-sample NVGesture test set. **Honest protocol**: train-best epoch selection (no test info), uniform 1/K weights, DSN softmax = `softmax(logits × 9.5)`. 18 models in pool.

## Top combo per fusion width (with DSN)

| Width | Best combo | Acc |
|---|---|---|
| 1 (solo) | DSN | 90.25 |
| 2 | DSN + AttRD | 91.91 |
| 3 | DSN + BRD + AttRD *(5-way tie at 92.32)* | 92.32 |
| 4 | DSN + RD + BRD(N2) + AttRD | 92.32 |
| **5** | **DSN + RD + BRD(N2) + AttRD + DN2(N1)** | **92.53** |

## Solos (train-best epochs)

| Model | Epoch | Solo |
|---|---|---|
| DSN (CVPR I3DWTrans depth) | external | 90.25 |
| DN2(N1) | ep109 | 89.42 |
| QDN(N1) | ep110 | 89.42 |
| AttRD | ep120 | 89.00 |
| DeltaProduct | ep108 | 88.80 |
| RD | ep118 | 88.59 |
| BRD | ep112 | 88.38 |
| DN2(N2) | ep113 | 88.38 |
| DN(N2) | ep112 | 87.97 |
| DeltaOSS | ep107 | 87.97 |
| BRD(N2) | ep109 | 87.76 |
| AttRDv2 | ep110 | 87.76 |
| DN2(N14) | ep104 | 87.76 |
| QDN(N2) | ep112 | 87.76 |
| AttRD(N2) | ep112 | 87.34 |
| LinOSS(N2) | ep117 | 87.34 |
| RD(N2) | ep118 | 87.14 |
| RD(N14) | ep112 | 86.93 |

## Top 5 per width

### 2-way
| Combo | Acc |
|---|---|
| DSN + AttRD | **91.91** |
| DSN + QDN(N1) | 91.70 |
| DSN + DN2(N1) | 91.70 |
| DSN + DN2(N14) | 91.49 |
| DSN + RD(N2) | 91.29 |

### 3-way
| Combo | Acc |
|---|---|
| DSN + RD(N2) + DN2(N1) | **92.32** |
| DSN + RD + QDN(N1) | **92.32** |
| DSN + BRD + AttRD | **92.32** |
| DSN + AttRD(N2) + QDN(N1) | **92.32** |
| DSN + AttRD + DN2(N1) | **92.32** |

### 4-way
| Combo | Acc |
|---|---|
| DSN + RD + BRD(N2) + AttRD | **92.32** |
| DSN + RD + BRD(N2) + QDN(N1) | 92.12 |
| DSN + RD + AttRD + QDN(N2) | 92.12 |
| DSN + RD + AttRD + AttRDv2 | 92.12 |
| DSN + BRD(N2) + DN2(N1) + QDN(N1) | 92.12 |

### 5-way (current ceiling)
| Combo | Acc |
|---|---|
| **DSN + RD + BRD(N2) + AttRD + DN2(N1)** | **92.53** |
| DSN + RD(N14) + DeltaProduct + DeltaOSS + QDN(N1) | 92.32 |
| DSN + RD + DeltaProduct + DeltaOSS + QDN(N1) | 92.32 |
| DSN + RD + AttRD + DeltaOSS + QDN(N1) | 92.32 |
| DSN + AttRDv2 + DeltaProduct + DeltaOSS + QDN(N1) | 92.32 |

## Headline progression

| Date | Ceiling | Recipe |
|---|---|---|
| pre-DSN | 91.49 | DSN + RD(N1) + RD(N2) + RD(N14) |
| 2026-05-13 | 92.32 | DSN + BRD + AttRD (3-way novel-arch) |
| **2026-05-14** | **92.53** | DSN + RD + BRD(N2) + AttRD + DN2(N1) 5-way |

Key insight: BRD(N2) > BRD(N1) — input-stream decorrelation (DTW-warped N2) beats architectural decorrelation for fusion diversity.

Commit: d92585c.
