# GB1 fitness prediction

Small, reproducible learning project. With only a few thousand measured GB1 variants, can simple models
predict the fitness of others and pick a better next batch than random sampling?

- Data: GB1 variants with 4 mutated positions (39, 40, 41, 54; reference VDGV), 149,361 measured variants.
- X = the 4 amino acids (one-hot, 80 features); y = continuous Fitness (regression, no binarisation).
- Models: training-mean baseline, one-hot + Ridge, small PyTorch MLP (80-64-64-1), 3 seeds.
- Split (random, disjoint): train 2,000 / val 2,000 / test 5,000 / candidate pool 20,000.
  Validation picks settings; test is only reported.

## Results (split seed 0, test set)

| Method | MAE | RMSE | Spearman |
|---|---|---|---|
| Training-mean baseline | 0.130 | 0.381 | undefined (constant prediction) |
| One-hot + Ridge | 0.172 | 0.349 | 0.499 |
| MLP (3-seed mean) | 0.062 | 0.230 | 0.461 |

Fitness is extremely right-skewed (median 0.0034, mean 0.08, max 8.76; about 20% exactly 0), so MAE alone is
misleading: Ridge has a worse MAE than the mean baseline but a reasonable ranking.

One retrospective candidate round (pick 20 from the 20,000-variant pool by predicted score, then reveal;
"hit" = true fitness in the pool's top 1%, threshold about 2.03; scoring fixed before looking):

| Strategy | Mean fitness of 20 picks | Hits |
|---|---|---|
| Ridge top 20 | 2.85 | 13 |
| MLP top 20 | 3.73 | 19 |
| Random 20 (5,000 draws) | 0.074 | 0.19 |
| Oracle ceiling | 5.06 | 20 |

Across split seeds 0 / 1 / 2, MLP hits were 19 / 15 / 17 and Ridge hits 13 / 10 / 4; the MLP beat the
random draws every time. Disagreement among the 3 MLP seeds correlates with absolute error
(Spearman about 0.35-0.40), a partial proxy for uncertainty, not a calibrated interval.

## Limitations
- Random split only; no test on more distant sequences or other proteins.
- One protein, four positions; a retrospective simulation, not a new wet-lab round.
- Training size fixed at 2,000; no hyperparameter search. Why the MLP beats Ridge (possible
  interactions between positions) is a hypothesis, not tested.
- Fitness 0 can reflect low reads or a measurement floor, not an exact biological value.

## Reproduce
```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
# put four_mutations_full_data.csv in data/raw/ (see below)
.venv/Scripts/python src/pipeline.py          # SEED=1 for another split
.venv/Scripts/python src/build_notebook.py    # builds gb1_fitness_prediction.ipynb
```
Notebooks:
- `GB1_Interview_Walkthrough.ipynb`: the guided, English walkthrough (plain-language explanations, interview checks with hidden answers, all figures, a three-split robustness check). Run All takes about 1-5 minutes on a CPU; install Jupyter first (`pip install jupyterlab`). Rebuild it with `python src/build_walkthrough.py`.
- `gb1_fitness_prediction.ipynb`: the compact pipeline version.

Outputs: `outputs/`.

## Data and licence
- Wu et al. (2016), *Adaptation in protein fitness landscapes is facilitated by indirect paths*, eLife 5:e16965 (data: CC BY 4.0).
- Processed version: FLIP benchmark (https://github.com/J-SNACKKB/FLIP), `splits/gb1/four_mutations_full_data.csv.zip`; modified files marked AFL-3.0.
- Data files are not committed; download the zip above and unzip into `data/raw/`.
