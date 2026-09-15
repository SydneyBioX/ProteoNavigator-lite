# ProteoNavigator Lite

ProteoNavigator Lite is a local-first Chainlit agent for building, reviewing,
and executing cell-annotation workflows with the Python `cytogater` package.

## Current capabilities

The current release supports:

- `.h5ad`, cells-by-markers `.csv`, and optional SpatialExperiment/SCE `.rds` input;
- wide `Populations + marker columns` and long Python lineage CSV formats;
- LLM-assisted lineage-table creation and revision;
- deterministic marker/panel validation;
- reviewable, versioned annotation plans and an explicit approval gate;
- executable soft- and tree-gating workflows;
- ordered gating-to-weighted-kNN, Random Forest, hierarchical kNN, neural-network,
  and hierarchical-kNN-to-RF-rescue workflows;
- explicit core selection, model training, and prediction stages that can be run
  and inspected independently;
- reusable trained-model artifacts, allowing prediction thresholds to be revised
  without retraining;
- candidate/cleaned core-cell counts, a per-cell-type core barplot, and spatial
  views after intermediate stages;
- downloadable annotated `.h5ad` and label-count CSV outputs;
- a reusable spatial canvas for coordinate, cell-type, and uncertainty views;
- OpenAI and Google Gemini models with user-entered model names.

Plans contain an ordered list of stages. Classifiers are never treated as
standalone methods when no reference labels were supplied: gating generates a
high-confidence reference before kNN, RF, hierarchical kNN, or neural-network
training. A hierarchical-plus-RF-rescue plan trains RF on confident hierarchical
labels and predicts only cells left `Unassigned`.

## Development install

The existing `cytogater-python` conda environment already contains cytoGater and
the agent dependencies on this workstation:

```bash
conda activate cytogater-python
cd /users/stgrad/lijiay/PRJ-cytogater/20260803-create_agent-LY/ProteoNavigator-lite
python -m pip install -e .
```

Because cytoGater is installed in the same environment, the agent imports that
installed package. For active development, ensure it is editable:

```bash
python -m pip install -e /users/stgrad/lijiay/PRJ-cytogater/cytoGateR/python
```

All cytoGater calls are isolated in `src/proteonavigator_lite/backend.py`, so an
API change requires one adapter update rather than changes to prompts and tools.

## Quick start: run the agent

Copy and run these commands on the machine where the project is stored:

```bash
conda activate cytogater-python
cd /users/stgrad/lijiay/PRJ-cytogater/20260803-create_agent-LY/ProteoNavigator-lite
python -m pip install -e .
DEBUG=false chainlit run app.py --host 127.0.0.1 --port 8000
```

Chainlit will print:

```text
Your app is available at http://localhost:8000
```

Open `http://localhost:8000` in a browser. The first screen asks you to:

1. choose `Google Gemini` or `OpenAI`;
2. enter a model name, such as `gemini-2.5-flash` or `gpt-4.1-mini`;
3. enter the corresponding API key;
4. press **Start session**.

You only need `python -m pip install -e .` the first time, or after changing the
package configuration. On later runs:

```bash
conda activate cytogater-python
cd /users/stgrad/lijiay/PRJ-cytogater/20260803-create_agent-LY/ProteoNavigator-lite
DEBUG=false chainlit run app.py --host 127.0.0.1 --port 8000
```

Alternatively, from the project directory:

```bash
make run
```

### Running on a remote cluster

If Chainlit is running on a remote server, create an SSH tunnel from a terminal
on your own computer:

```bash
ssh -L 8000:127.0.0.1:8000 YOUR_USERNAME@YOUR_CLUSTER_HOST
```

Inside that SSH session, run:

```bash
conda activate cytogater-python
cd /users/stgrad/lijiay/PRJ-cytogater/20260803-create_agent-LY/ProteoNavigator-lite
DEBUG=false chainlit run app.py --host 127.0.0.1 --port 8000
```

Then open `http://localhost:8000` on your own computer. If port 8000 is already
occupied, replace it with another port, such as 8001, in both commands.

### First example

Open the displayed local URL, choose Google Gemini or OpenAI, enter the exact
model name and API key, then upload a dataset. The API key stays in Chainlit
session memory and is not written to the repository or chat messages.

You can upload one of the existing pairs, for example:

```text
../data/datasets_h5ad/data_immucan.h5ad
../data/lineage_tables_csv/data_immucan.csv
```

Example requests:

```text
Inspect this dataset and explain which cell types the uploaded lineage can identify.
```

```text
Review the lineage table. Suggest corrections, but do not save changes yet.
```

```text
Create one ordered annotation plan that runs tree gating and then Random Forest.
```

After the agent creates a plan, use **Approve and run** to execute the exact
recorded lineage revision and parameters.

For a stepwise workflow, ask for one stage at a time:

```text
Run tree gating only.
Select core cells with quantile 0.98.
Train random forest only, then let me inspect the cleaned core cells.
Predict with the saved random-forest model at threshold 0.7.
Predict again with the same model at threshold 0.5.
Calculate uncertainty on this final prediction.
```

After core selection/training, the result includes a core-count CSV, a grouped
barplot of candidate versus cleaned core cells per cell type, and a spatial view
colored by the current core label. After prediction, the completion message
reports assigned/unassigned counts, newly assigned cells, and confidence
statistics. You can repeat a matching `predict_*` stage with new parameters or
train another model from the saved intermediate `.h5ad`.

The approval card lists stages in execution order and expands default values,
for example:

```text
1. tree_gating — max_depth, min_cells, min_score, cutoff method, workers, ...
2. random_forest — reference quantile, trees, CV folds, repeats, thresholds, seed, ...
```

While cytoGater is running, the message updates every five seconds with elapsed
time, the number of cells, and the current pipeline phase (for example, tree
gating, reference construction, Random Forest, or output writing). It is not a
percentage within a phase because the current cytoGater API does not expose that
detail. The same message changes to either **completed** with downloads or
**failed** with the error details when execution finishes.

## Input contracts

### AnnData

Cells must be rows and markers columns. If `layers["exprs"]` is absent, the
loader copies `X` into that layer for compatibility with cytoGater.

### Expression CSV

The first column contains cell identifiers. Every remaining column must be a
numeric marker expression column.

### RDS

RDS support invokes `Rscript` and requires `SingleCellExperiment`; spatial
objects additionally require `SpatialExperiment`. The converter chooses the
`exprs` assay when present, otherwise the first assay. Prefer `.h5ad` for the
simplest portable installation.

### Lineage CSV

Either the existing wide format:

```text
Populations,CD3,CD20
T cell,1,-1
B cell,-1,1
```

or a long format:

```text
cell_type,pos_markers,neg_markers
T cell,"CD3","CD20"
B cell,"CD20","CD3"
```

## Tests

```bash
conda run -n cytogater-python pytest -q
```

Runtime uploads, lineage revisions, plans, outputs, and an API-key-free
`conversation.jsonl` transcript live under `runtime/<session-id>/`. Session
directories are retained when the chat ends so that runs remain reproducible.
Back up or remove old session directories according to your local storage policy.

When `obsm["spatial"]` is available (including converted SpatialExperiment or
SingleCellExperiment RDS inputs), the app displays a spatial viewer after upload. When
the agent asks which annotation method to use, the same question also asks whether to
add `calculate_uncertainty` as the final stage; it is never added silently. After
annotation, the panel always gains **Cell type** and gains
**Uncertainty** only when that optional stage was approved. With that stage, the
downloaded AnnData stores `annotation_label`, `entropy`, `gini_impurity`,
`margin_uncertainty`, and `spatial_discordance`. The four uncertainty fields come
directly from cytoGater's `calculate_uncertainty()` and use a complete,
row-normalized probability matrix. Weighted kNN and RF use their combined reference
and prediction probabilities; workflows without complete downstream probabilities
use gating probabilities and record that source in AnnData metadata. Viewer payloads
are bounded to 30,000 deterministic points; analysis always uses all cells.
For multi-sample datasets, the viewer shows one sample at a time through a sample
selector. It does not overlay samples or create hundreds of simultaneous panels.
The viewer uses Chainlit's side workspace, producing a two-column viewer-and-chat
layout. The same side element is updated after annotation instead of adding a second
viewer to the conversation.

Analysis can be approved as one complete ordered plan or performed step by step. In
stepwise mode, each completed annotated `.h5ad` becomes the input to the next plan, so
you can run gating, inspect it, request RF or kNN with revised parameters, inspect that
result, and then request `calculate_uncertainty` without rerunning earlier stages. The
probability matrix required by later stages is retained inside each output AnnData.

Core-reference construction can also be an explicit `select_core_cells` stage, using
either per-class probability quantiles or MAD cutoffs. Its `cutoff_label` result is saved
and shown in the spatial viewer before a later RF, kNN, hierarchy, or neural-network
continuation. Tree-gating models are saved as JSON; after a tree run, ask the agent to
list available tree cell types or render one (for example, "show the Tumor tree plot").
# ProteoNavigator-lite
