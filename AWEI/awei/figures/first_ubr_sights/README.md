# First Look at UBR
Result dir: `"/home/benjamin/Dokumente/code/tmp/DAC-BO/AWEI/awei_runs/2023-01-24/15-48-18"`
ndim: 2
nfunctions: 25
ninstances: 1
policies: ei, pi, explore, eipi_s50

## Function Families

### Separable functions
1. Sphere Function
2. Ellipsoidal Function
3. Rastrigin Function
4. Büche-Rastrigin Function
5. Linear Slope

### Functions with low or moderate conditioning
6. Attractive Sector Function
7. Step Ellipsoidal Function
8. Rosenbrock Function, original
9. Rosenbrock Function, rotated

### Functions with high conditioning and unimodal
10. Ellipsoidal Function
11. Discus Function
12. Bent Cigar Function
13. Sharp Ridge Function
14. Different Powers Function

### Multi-model functions with adequate global structure
15. Rastrigin Function
16. Weierstrass Function
17. Schaffers F7 Function
18. Schaffers F5 Function, moderately ill-conditioned
19. Composite Griewank-Rosenbrock Function F8F2

### Multi-model funcitons with weak global structure
20. Schwefel Function
21. Gallagher's Gaussian 101-me Peaks Function
22. Gallagher's Gaussian 21-hi Peaks Function
23. Katsuura Function
24. Lunacek bi-Rastrigin Function


## Observations
- The worse the reward, the lower UBR
- Really large scales at times
- For each schedule UBR is different
- --> UBR carries signal

I have the feeling that I missed an implementation detail. What did they do about the scales? What is the true regret?

--> Change: Add top p (50%) selection of evaluated configs as the paper states (Section 4)

Policy should be function of remaining steps and UBR

UBR: Lower = better?

Rules:
- Gradient  magnitude high: Stay
- Gradient low: change
  - gradients positive: direction 1
  - gradients negative: direciton 2

Difficulties:
