# Cosine Annealing Learning Rate Schedule

Generated using `get_lr_cosine_schedule()` from `cs336_basics/optimizer.py` with:

- `max_learning_rate = 1.0`
- `min_learning_rate = 0.1`
- `warmup_iters = 10`
- `cosine_cycle_iters = 100`

```mermaid
xychart-beta
    title "Cosine schedule with linear warmup"
    x-axis "iteration" [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    y-axis "learning rate" 0.0 --> 1.0
    line [0.0000, 0.5000, 1.0000, 0.9932, 0.9729, 0.9397, 0.8947, 0.8393, 0.7750, 0.7039, 0.6281, 0.5500, 0.4719, 0.3961, 0.3250, 0.2607, 0.2053, 0.1603, 0.1271, 0.1068, 0.1000]
    line [0.0000, 0.5000, 1.0000, 0.9932, 0.9729, 0.9397, 0.8947, 0.8393, 0.7750, 0.7039, 0.6281, 0.5500, 0.4719, 0.3961, 0.3250, 0.2607, 0.2053, 0.1603, 0.1271, 0.1068, 0.1000]
```

| iteration | learning rate |
| ---: | ---: |
| 0 | 0.000000 |
| 5 | 0.500000 |
| 10 | 1.000000 |
| 15 | 0.993163 |
| 20 | 0.972862 |
| 25 | 0.939711 |
| 30 | 0.894720 |
| 35 | 0.839254 |
| 40 | 0.775000 |
| 45 | 0.703909 |
| 50 | 0.628142 |
| 55 | 0.550000 |
| 60 | 0.471858 |
| 65 | 0.396091 |
| 70 | 0.325000 |
| 75 | 0.260746 |
| 80 | 0.205280 |
| 85 | 0.160289 |
| 90 | 0.127138 |
| 95 | 0.106837 |
| 100 | 0.100000 |

The schedule linearly warms up from `0.0` to `1.0` over the first `10` iterations, then cosine-anneals down to `0.1` by iteration `100`.
