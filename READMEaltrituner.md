# Implementazioni Altri Tuner

Questo file descrive le implementazioni aggiunte per confrontare EDGE-DNN Tuner con tuner esterni sullo stesso spazio di ricerca.

I tuner implementati sono:

- BANANAS
- FlexiBO

L'obiettivo principale e' fare esperimenti comparabili usando lo stesso spazio di ricerca del progetto, cioe' lo spazio piu' ampio ottenuto con:

```python
cfg.opt = "RS"
```

Entrambi i runner costruiscono lo spazio tramite:

```python
base_space = search_space().search_sp(max_block=ctrl.max_conv, max_dense=ctrl.max_fc)
```

e quindi partono dallo stesso `components/search_space.py` usato dal progetto.

## Spazio Di Ricerca RS

Lo spazio RS include sia scelte architetturali sia iperparametri di training. In particolare contiene variabili come:

```text
activation
optimizer
learning_rate
batch_size
num_neurons
dr_f
data_augmentation
reg_l2
skip_connection
unit_c1
unit_c2
new_fc_1 ... new_fc_10
```

Rispetto agli spazi originali di molti tuner NAS, questo e' uno spazio misto:

- categorico
- intero
- continuo
- booleano
- architetturale
- di training

Questa e' la ragione per cui i tuner non vengono usati "as-is" dai rispettivi repository originali, ma adattati per interrogare lo stesso spazio RS del progetto.

## BANANAS

File principale:

```text
bananas_runner.py
```

Dipendenza originale:

```text
othertunerdependencies/bananas/naszilla
```

Questa directory e' un git submodule del repository NASzilla, che contiene
l'implementazione originale di BANANAS usata come riferimento.

Script Slurm:

```text
submit_bananas_cifar_rolling_array_slurm.sh
submit_bananas_cifar_cpu_controller_slurm.sh
```

### Idea Implementata

BANANAS e' implementato come adapter verso il codice originale NASzilla/BANANAS:

1. il runner espone lo spazio RS del progetto con una interfaccia compatibile con BANANAS;
2. ogni configurazione RS viene trattata come una "architettura" NASzilla;
3. l'encoding e' prodotto dall'adapter;
4. le mutazioni sono definite dall'adapter sullo spazio RS;
5. il neural predictor e le acquisition function arrivano dal submodule NASzilla;
6. la valutazione reale e' demandata al controller del progetto.

La funzione obiettivo e' quella restituita dal controller:

```python
score = ObjectiveWrapper(base_space, ctrl).objective(point)
```

Quindi BANANAS non usa una funzione obiettivo propria. Usa la stessa pipeline di valutazione del progetto.

### Objective

BANANAS minimizza lo score restituito da:

```python
controller.training(...)
```

Nel caso senza moduli, il controller usa sostanzialmente:

```python
score = -accuracy
```

quindi minimizzare lo score equivale a massimizzare la validation accuracy.

Nel caso usato negli esperimenti con:

```bash
--mod_list flops_module
```

il controller applica anche il controllo FLOPs/params. Se il modello viola i vincoli, assegna:

```python
score = 1e10
```

Se il modello e' valido, lo score puo' includere il contributo dei moduli del progetto.

### Encoding

Ogni punto dello spazio RS viene codificato cosi':

- `Categorical`: one-hot encoding;
- `Integer`: normalizzazione lineare in `[0, 1]`;
- `Real`: normalizzazione lineare in `[0, 1]`.

Questo encoding e' usato per addestrare il predittore neurale.

### Predittore Neurale

Il predittore neurale viene da NASzilla:

```python
from naszilla.meta_neural_net import MetaNeuralnet
```

Di default usa la configurazione tipica NASzilla:

```text
num_layers = 10
layer_width = 20
loss = mae
lr = 0.01
epochs = 200
```

Parametri principali:

```bash
--ensemble_size 5
--predictor_epochs 200
--metann_num_layers 10
--metann_layer_width 20
--metann_lr 0.01
--metann_loss mae
```

### Acquisition

L'acquisition viene da NASzilla:

```python
from naszilla.acquisition_functions import acq_fn
```

Il default e':

```bash
--explore_type its
```

Sono disponibili le modalita' supportate da NASzilla:

```bash
ucb
ei
pi
ts
percentile
mean
confidence
its
```

### Mutazioni

La prima versione locale usava un candidate pool random. E' stata salvata con suffisso `_wrong` e non va usata per gli esperimenti finali.

La versione attuale usa NASzilla per predittore e acquisition, ma mantiene un adapter di mutazione sullo spazio RS, perche' NASzilla originale sa mutare celle NASBench, non configurazioni miste DNN-Tuner.

Ora, dopo le valutazioni iniziali random, il candidate pool e' costruito principalmente mutando i migliori punti gia' valutati.

Parametri:

```bash
--mutation_parents 10
--mutation_attempts 100
--random_candidate_fraction 0.10
```

La logica e':

1. ordina le configurazioni gia' valutate per score crescente;
2. prende i migliori `mutation_parents`;
3. sceglie un parent;
4. muta una sola dimensione;
5. scarta il candidato se gia' visto.

Mutazioni implementate:

- `Categorical`: cambia categoria scegliendone una diversa;
- booleani: sono trattati come categorici, quindi vengono flippati;
- `Integer`: con probabilita' 0.8 fa uno step locale, altrimenti campiona random nel range;
- `Real`: con probabilita' 0.8 applica una perturbazione gaussiana locale, altrimenti campiona random nel range.

Per gli interi, lo step locale e' circa il 10% del range:

```python
step = max(1, round((high - low) * 0.10))
```

Per i reali, la deviazione standard della perturbazione e' circa il 10% del range:

```python
sigma = (high - low) * 0.10
```

Una piccola frazione di candidati resta random (`random_candidate_fraction`) per mantenere esplorazione globale.

### Fedelta' Rispetto A BANANAS Originale

La formulazione e' piu' difendibile della reimplementazione locale perche' usa direttamente componenti NASzilla originali:

- `MetaNeuralnet`;
- `acq_fn`;
- schema BANANAS random-init + predictor ensemble + acquisition + mutation.

Resta comunque un adapter, non una chiamata diretta a `naszilla.nas_algorithms.bananas(...)`, perche':

- NASzilla si aspetta search space NASBench con metodi come `generate_random_dataset`, `get_candidates`, `query_arch`, `mutate_arch`;
- lo spazio RS non e' una cella NASBench ma una configurazione mista;
- la valutazione e' `controller.training(...)`;
- sono stati aggiunti resume e chunk Slurm.

Quindi la descrizione corretta e':

```text
NASzilla/BANANAS with an EDGE-DNN Tuner RS search-space adapter.
```

### File Con Suffisso `_wrong`

I file:

```text
bananas_runner_wrong.py
submit_bananas_*_wrong.sh
```

sono backup della prima implementazione locale. Restano nel branch per tracciabilita', ma non sono la versione da usare per gli esperimenti finali.

## FlexiBO

File principale:

```text
flexibo_runner.py
```

Dipendenza originale:

```text
othertunerdependencies/flexibo/FlexiBO
```

Questa directory e' un git submodule del repository FlexiBO originale.

Script Slurm:

```text
submit_flexibo_cifar_rolling_array_slurm.sh
submit_flexibo_cifar_cpu_controller_slurm.sh
```

### Idea Implementata

FlexiBO e' implementato come baseline Bayesian optimization multi-obiettivo e cost-aware.

Gli obiettivi sono:

```text
O1 = validation error = 1 - validation accuracy
O2 = normalized FLOPs = FLOPs / flops_scale
```

Questa scelta rende l'adattamento piu' vicino all'idea originale di FlexiBO, che e' pensato per ottimizzazione multi-obiettivo con costi diversi di valutazione.

Nel nostro caso:

- accuracy costa molto, perche' richiede training;
- FLOPs costa poco, perche' richiede solo build del modello.

La parte piu' specifica di FlexiBO, cioe' la selezione del prossimo
`sample, objective` tramite frontiere pessimistiche/ottimistiche e rapporto
`delta-volume / cost`, viene dal codice originale:

```python
from src.sampling import Sampling
from src.utils import Utils
```

Anche i wrapper dei surrogate vengono dal repository originale:

```python
from src.surrogate_model import GPSurrogateModel
from src.surrogate_model import RFSurrogateModel
```

Il runner locale costruisce le regioni di incertezza a partire dai surrogate
addestrati sui punti RS e poi chiama il sampling originale. Se il codice
originale non riesce su un batch di candidati RS, il runner usa un fallback
locale basato su incertezza/costo per non interrompere le run lunghe.

### Objective E Costi

Gli obiettivi sono minimizzati.

Per l'accuracy:

```python
error = 1.0 - accuracy
```

Per i FLOPs:

```python
flops_norm = flops / flops_scale
```

Di default:

```bash
--accuracy_cost 1.0
--flops_cost 0.05
```

La scala FLOPs di default e' `--flops_th`, cioe':

```bash
--flops_th 150000000
```

oppure puo' essere impostata manualmente:

```bash
--flops_scale <valore>
```

### Valutazione FLOPs-Only

FlexiBO puo' decidere di misurare solo i FLOPs di una configurazione.

In quel caso il runner:

1. converte il punto dello spazio in dizionario di parametri;
2. imposta data augmentation, L2 e skip connection sul controller;
3. costruisce il modello con:

```python
ctrl.nn.build_network(params, ctrl.layer_x_block)
```

4. legge:

```python
ctrl.nn.flops
ctrl.nn.nparams
```

5. salva la misura in:

```text
algorithm_logs/flexibo_flops_only_report.txt
```

Questa valutazione non consuma budget `--eval`, perche' non allena il modello.

### Valutazione Accuracy

Quando FlexiBO decide di misurare accuracy, il runner chiama il training completo:

```python
score = ctrl.training(params)
```

Da questo ricava:

```python
accuracy = ctrl.scoreNN[1]
error = 1.0 - accuracy
flops = ctrl.nn.flops
params = ctrl.nn.nparams
```

Questa valutazione consuma una unita' del budget `--eval`.

Quindi in FlexiBO:

```bash
--eval 1000
```

significa 1000 valutazioni di accuracy/training, non 1000 eventi totali. Gli eventi FLOPs-only sono extra e costano poco.

### Surrogate

FlexiBO mantiene due modelli surrogati separati:

- uno per `error`;
- uno per `flops_norm`.

Sono disponibili due surrogate:

```bash
--surrogate GP
--surrogate RF
```

Con `GP`, usa:

```python
GaussianProcessRegressor
Matern(nu=2.5) + WhiteKernel
```

Con `RF`, usa:

```python
RandomForestRegressor
```

### Candidate Pool

FlexiBO genera un candidate pool campionando dallo spazio RS.

Il pool include:

- punti gia' parzialmente misurati, cioe' con solo accuracy o solo FLOPs;
- nuovi punti campionati dallo spazio.

Parametro:

```bash
--candidate_pool 512
```

### Scelta Del Prossimo Obiettivo

FlexiBO predice per ogni candidato:

```text
mean(error), std(error)
mean(flops), std(flops)
```

Costruisce regioni pessimistiche, medie e ottimistiche. Poiche' gli obiettivi
EDGE sono da minimizzare, mentre il codice originale FlexiBO lavora
operativamente su assi da massimizzare, il runner passa al codice originale le
utilita' negative:

```python
utility_error = -error
utility_flops = -flops_norm
```

La scelta primaria viene poi fatta da `Sampling.determine_next_sample(...)` del
repository originale FlexiBO, che valuta la riduzione della regione Pareto
incerta rispetto al costo dell'obiettivo. Quindi:

- se l'incertezza su FLOPs e' alta e il costo e' basso, puo' scegliere `flops`;
- se l'incertezza su accuracy e' importante, puo' scegliere `accuracy`;
- all'inizio usa `both`, cioe' misura accuracy e FLOPs insieme, per costruire un training set iniziale.

Parametro:

```bash
--beta 1.0
```

### History

FlexiBO salva:

```text
algorithm_logs/flexibo_history.csv
```

Colonne:

```text
event
accuracy_eval
objective
error
accuracy
flops_norm
flops
params
score
elapsed_sec
params_json
```

`event` conta ogni evento FlexiBO.

`accuracy_eval` conta solo le valutazioni con training.

`objective` puo' essere:

```text
both
accuracy
flops
```

### Resume

FlexiBO supporta:

```bash
--resume
--max_new_evals
```

Il resume ricarica `flexibo_history.csv`.

Importante: la nuova implementazione multi-obiettivo non e' compatibile con eventuali vecchie history FlexiBO single-objective. Se trova una history vecchia, il runner fallisce e chiede di usare una nuova directory risultati.

Usare quindi una directory nuova, ad esempio:

```text
results_FLEXIBO_mo_controller
```

### Fedelta' Rispetto A FlexiBO Originale

Questa implementazione e' piu' fedele della prima versione single-objective perche':

- usa due obiettivi;
- associa costi diversi agli obiettivi;
- puo' scegliere quale obiettivo misurare;
- usa FLOPs come obiettivo economico;
- usa accuracy/error come obiettivo costoso;
- usa `GPSurrogateModel` e `RFSurrogateModel` dal repository originale FlexiBO;
- usa `Sampling` e `Utils` dal repository originale FlexiBO per frontiere pessimistiche/ottimistiche e scelta `delta-volume / cost`.

Resta comunque un adattamento, non una chiamata diretta a `RunFlexiBO.py`, perche':

- lo spazio e' lo spazio RS di EDGE-DNN Tuner, non il loro spazio hardware/network da YAML;
- l'evaluator e' quello del progetto;
- i surrogate vengono addestrati sui punti RS del progetto;
- il deployment/hardware measurement originale e' sostituito da FLOPs calcolati dal progetto.

Descrizione corretta:

```text
FlexiBO adapted to the EDGE-DNN Tuner RS search space, using original FlexiBO sampling over validation error and FLOPs objectives.
```

La prima versione locale/single-objective non va usata per gli esperimenti
finali.

I backup della versione precedente sono:

```text
flexibo_runner_wrong.py
submit_flexibo_*_wrong.sh
```

## Slurm

Dopo un pull sul cluster, aggiornare anche i submodule:

```bash
git submodule update --init --recursive
```

Per entrambi i tuner sono disponibili due livelli di script:

```text
*_rolling_array_slurm.sh
*_cpu_controller_slurm.sh
```

Il modello consigliato e' usare il controller CPU:

- il controller gira su partizione CPU;
- sottomette un solo array GPU per chunk;
- sottomette il controller successivo con dipendenza;
- evita di tenere processi sul login node;
- evita di intasare `squeue` con decine di array gia' pendenti.

Le partizioni GPU ammesse dagli script sono:

```text
gpu_H100
gpu_H100_partitioned
gpu_L40S
```

Per evitare di prendere l'H100 intera, usare:

```bash
PARTITION=gpu_H100_partitioned
```

## Esempio BANANAS

```bash
JOB_SETUP="module load cuda/12.2" \
CONTROLLER_PARTITION=cpu_amd_zen4 \
CONTROLLER_MEMORY=256M \
CONTROLLER_TIME=00:02:00 \
CONTROLLER_DEPENDENCY=afterany \
PARTITION=gpu_H100_partitioned \
MEMORY=64G \
TOTAL_EVALS=1000 \
CHUNK_EVALS=25 \
EPOCHS=100 \
TIME_LIMIT=10:00:00 \
MAX_PARALLEL=3 \
START_CHUNK=1 \
MUTATION_PARENTS=10 \
MUTATION_ATTEMPTS=100 \
RANDOM_CANDIDATE_FRACTION=0.10 \
RESULTS_DIR=results_BANANAS_mutation_controller \
bash submit_bananas_cifar_cpu_controller_slurm.sh
```

## Esempio FlexiBO Multi-Objective

```bash
JOB_SETUP="module load cuda/12.2" \
CONTROLLER_PARTITION=cpu_amd_zen4 \
CONTROLLER_MEMORY=256M \
CONTROLLER_TIME=00:02:00 \
CONTROLLER_DEPENDENCY=afterany \
PARTITION=gpu_H100_partitioned \
MEMORY=64G \
TOTAL_EVALS=1000 \
CHUNK_EVALS=25 \
EPOCHS=100 \
TIME_LIMIT=10:00:00 \
MAX_PARALLEL=3 \
START_CHUNK=1 \
SURROGATE=GP \
BETA=1.0 \
ACCURACY_COST=1.0 \
FLOPS_COST=0.05 \
RESULTS_DIR=results_FLEXIBO_mo_controller \
bash submit_flexibo_cifar_cpu_controller_slurm.sh
```

## Controllo Avanzamento

BANANAS:

```bash
find results_BANANAS_mutation_controller -name bananas_history.csv \
  -exec sh -c 'printf "%-110s %s eval\n" "$1" "$(( $(wc -l < "$1") - 1 ))"' sh {} \; | sort
```

FlexiBO:

```bash
find results_FLEXIBO_mo_controller -name flexibo_history.csv \
  -exec sh -c 'printf "%-110s %s accuracy eval\n" "$1" "$(awk -F, '\''NR > 1 && ($3 == "accuracy" || $3 == "both") {c++} END {print c + 0}'\'' "$1")"' sh {} \; | sort
```

## Note Metodologiche

Il confronto con EDGE-DNN Tuner e' pensato per rispondere a questa domanda:

```text
A parita' di spazio RS e pipeline di valutazione, come si comportano strategie di ricerca alternative?
```

Non va presentato come riproduzione esatta dei benchmark originali BANANAS/FlexiBO.

Formulazione consigliata:

```text
We adapted BANANAS and a multi-objective FlexiBO-style optimizer to the same RS search space and evaluation pipeline used by EDGE-DNN Tuner. This isolates the effect of the search strategy under a shared experimental setting.
```
