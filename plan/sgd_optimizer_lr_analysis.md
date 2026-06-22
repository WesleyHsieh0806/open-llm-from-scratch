| learning rate | iter 0 | iter 1 | iter 2 | iter 3 | iter 4 | iter 5 | iter 6 | iter 7 | iter 8 | iter 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1e1 | 26.2714 | 16.8137 | 12.3943 | 9.69725 | 7.85477 | 6.51251 | 5.49243 | 4.69344 | 4.05316 | 3.53075 |
| 1e2 | 26.2714 | 26.2714 | 4.50746 | 0.107874 | 1.15257e-16 | 1.28461e-18 | 4.32572e-20 | 2.57686e-21 | 2.2106e-22 | 2.45622e-23 |
| 1e3 | 26.2714 | 9483.98 | 1.63803e+06 | 1.82214e+08 | 1.47593e+10 | 9.31481e+11 | 4.78192e+13 | 2.05739e+15 | 7.58309e+16 | 2.43501e+18 |

With `lr=1e1`, the loss decays steadily but relatively gradually; with `lr=1e2`, it decays much faster and reaches essentially zero within a few iterations. With `lr=1e3`, the loss diverges, increasing rapidly over the 10 iterations.
