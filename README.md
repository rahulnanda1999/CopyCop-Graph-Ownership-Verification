# Graph Ownership Verification

## Setup

```bash
conda create -n graph_ownership_verification python=3.10
conda activate graph_ownership_verification
pip install -r requirements.txt
```

## Usage

All steps below are independent after Step 1 completes. Run them in order the first time; afterwards, any step can be re-run on its own since intermediate results are cached to disk.

### Step 1: Train models and find stationary points

This trains base models, stolen models, and computes stationary points for a dataset. Takes the longest to run. Results are saved as `StatPts_*` and `Z_*` files.

```python
import get_data, our_setup

citeseer = get_data.get_data_citeseer()
all_Z_citeseer = our_setup.do_all(
    dataset=citeseer, dataset_name='citeseer',
    split_size=2000, dim_h=8, dim_out=6,
    bm_loss_type='CE', bm_end_target='node', epochs=500
)
```

### Step 2: Aggregate results across all datasets

Loads all saved `Z_*` files and assembles a summary DataFrame. Only includes datasets whose Step 1 has completed.

```python
import our_setup

all_df = our_setup.look_at_all_results()
all_df.to_csv('AllResults.csv')
```

### Step 3: Evaluate detection (AUC and accuracy)

Computes AUC for distinguishing stolen vs independent models.

```python
all_out, all_auc, loss_and_score_have_sign_pos_corr = our_setup.check_alldf(
    all_df, threshold='mid', which_steals='bestloss', measure='pscore'
)
all_auc.to_csv('AUC_results.csv')
all_out.to_csv('AllOut.csv')
```

### Step 4: Test robustness to embedding transforms

Tests whether detection still works after applying transformations (rotations, scaling, nonlinearities, etc.) to the stolen model's embeddings.

```python
import transforms

all_df_pscores, all_df_pass, all_df_pass_allbms = transforms.do_all(citeseer=citeseer)
all_df_pscores.to_csv('AllDfPscores.csv')
all_df_pass.to_csv('AllDfPass.csv')
all_df_pass_allbms.to_csv('AllDfPassAllBms.csv')
```

### Step 5: Sensitivity analysis (regularization strength)

Measures how detection AUC varies with different lambda_reg values.

```python
import sensitivity

auc_sens = sensitivity.do_several_vary_lambdareg(all_df, citeseer=citeseer)
auc_sens.to_csv('VaryLambda.csv')
```

### Step 6: Stationary point diversity

Computes pairwise distances and cosine similarities between stationary points to verify they are diverse.

```python
import pickle, numpy as np
from pandas import Series

with open('StatPts_citeseer_bm_GCN8s1', 'rb') as file:
    this_pts = pickle.load(file)

z = [this_pts[i][j]['lo']
     for i in this_pts.keys()
     for j in list(this_pts[i].keys())[0:1]
     if this_pts[i][j]['lo'].shape[0] == 2]

dists = [np.linalg.norm(z[i] - z[j]) / np.sqrt(np.linalg.norm(z[i]) * np.linalg.norm(z[j]))
         for i in range(len(z)) for j in range(i + 1, len(z))]
Series(dists).to_csv('DISTS_citeseer_lo.csv')
```

### Step 7: Pscore distribution of independent models

Extracts pscores for independently trained (non-stolen) models, used to show they are separable from stolen models.

```python
tmp = all_df.swaplevel().sort_index().loc['pscore'].swaplevel(axis=1).sort_index(axis=1)
indep_cols = [x for x in tmp.columns.values
              if 'steal' not in x[0]
              and 'boot' not in x[0]
              and x[0].split('_')[-1] != f'{x[1]}s1']
tmp[indep_cols].T.to_csv('INDEP_MODEL_PSCORES.csv')
```

## Output files

| File | Produced by | Description |
|------|-------------|-------------|
| `StatPts_*` | Step 1 | Stationary points per base model |
| `Z_*` | Step 1 | Per-model evaluation metrics at stationary points |
| `*.weights.pth` | Step 1 | Trained model weights |
| `AllResults.csv` | Step 2 | Aggregated results across all datasets |
| `AUC_results.csv` | Step 3 | Detection AUC per base model and dataset |
| `AllOut.csv` | Step 3 | Per-model classification (stolen vs independent) |
| `AllDfPscores.csv` | Step 4 | Pscores under each embedding transform |
| `AllDfPass.csv` | Step 4 | Whether each transform passes detection |
| `AllDfPassAllBms.csv` | Step 4 | Pass rates relative to all independent base models |
| `VaryLambda.csv` | Step 5 | AUC vs regularization strength |
| `DISTS_citeseer_lo.csv` | Step 6 | Pairwise distances between stationary points |
| `INDEP_MODEL_PSCORES.csv` | Step 7 | Pscore distribution for independent models |