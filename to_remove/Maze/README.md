## TODO

1. [DONE] Fix some walls
   1. [DONE] [(1,1), (2,2), (0,4), (1,4),  (4,0), (4,1), (4,4), (4,5), (4,6), (5,4), (5, 5), (5, 6), (6,4), (6, 5), (6, 6)],
2. [DONE] Change failure probability to probability of wrong move
3. Compare
   1. Algorithm 1 in https://arxiv.org/pdf/2106.02847.pdf
   2. Explorative policy (6) in the draft, where
      1. pi*, V* are computed using the current estimate of the model
      2. pi*, V* are given by Q learning