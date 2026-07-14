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

## Mappa Dei File Del Branch

Questa sezione riassume cosa e' stato aggiunto o modificato nel branch
`RESCAALTRITUNER` rispetto al branch di partenza `master_show`.

### File Python Aggiunti

`bananas_runner.py`

Runner attuale per BANANAS. Collega il tuner allo spazio RS del progetto,
converte le configurazioni RS nel formato atteso da NASzilla/BANANAS,
implementa le mutazioni sullo spazio RS, usa il neural predictor e le
acquisition function vendorizzate da NASzilla, e delega la valutazione reale
al controller del progetto. Supporta resume/chunking per Slurm e scrive i log
in `algorithm_logs/bananas_history.csv`.

`flexibo_runner.py`

Runner attuale per FlexiBO. Collega FlexiBO allo stesso spazio RS e usa una
formulazione multi-objective con accuracy e FLOPs. Quando possibile misura solo
i FLOPs, mentre le valutazioni costose di accuracy passano dal controller del
progetto. Usa le parti vendorizzate di FlexiBO per sampling, surrogate model e
utility, e scrive i log in `algorithm_logs/flexibo_history.csv`.

`export_tuner_results_csv.py`

Script locale di post-processing. Legge i risultati scaricati dal cluster,
principalmente `results_BANANAS_naszilla_controller_cluster/` e
`results_FLEXIBO_controller_cluster/`, e genera:

- `tuner_results_summary.csv`
- `tuner_results_aggregates.csv`
- `tuner_results_paper_table.csv`

Il primo file contiene una riga per run. Il secondo contiene medie e statistiche
aggregate per tuner, dataset e seed. Il terzo contiene una tabella aggregata in
stile paper.

### Codice Vendorizzato Da Altre Repository

Le dipendenze usate per rendere le implementazioni piu' fedeli agli originali
sono state copiate dentro `othertunerdependencies/` invece di usare submodule.
Questo evita di dover eseguire `git submodule update --init --recursive` sul
cluster.

`othertunerdependencies/bananas/naszilla/naszilla/acquisition_functions.py`

Funzioni di acquisizione NASzilla/BANANAS usate per scegliere le candidate
successive dopo il fit del predictor.

`othertunerdependencies/bananas/naszilla/naszilla/meta_neural_net.py`

Implementazione del meta neural network/predictor usato da BANANAS per stimare
le performance delle architetture candidate.

`othertunerdependencies/bananas/naszilla/naszilla/__init__.py`

File di package Python per importare il codice NASzilla vendorizzato.

`othertunerdependencies/flexibo/FlexiBO/src/sampling.py`

Logica FlexiBO per campionare configurazioni e obiettivi da valutare.

`othertunerdependencies/flexibo/FlexiBO/src/surrogate_model.py`

Surrogate model FlexiBO, inclusi i wrapper per modelli probabilistici usati
nella scelta delle configurazioni.

`othertunerdependencies/flexibo/FlexiBO/src/utils.py`

Utility FlexiBO per gestione dei dati, candidati e calcoli ausiliari usati dal
runner.

`othertunerdependencies/flexibo/FlexiBO/src/__init__.py`

File di package Python per importare il codice FlexiBO vendorizzato.

### Script Slurm E Utility Aggiunti

`scarica.sh`

Scarica risultati e log dal cluster. Accetta:

- `remote`, che usa `fresca@copernico.unife.it`
- `lan`, che usa `fresca@copernico.endif.man`

Sincronizza i risultati BANANAS, i risultati FlexiBO e `slurm_logs/` in
directory locali con suffisso `_cluster`.

`submit_bananas_cifar_rolling_array_slurm.sh`

Script worker/array attuale per BANANAS. Esegue chunk successivi della stessa
run leggendo lo stato gia' presente nei log, quindi permette resume senza
perdere le valutazioni gia' completate.

`submit_bananas_cifar_cpu_controller_slurm.sh`

Controller leggero su partizione CPU. Sottomette un array GPU BANANAS alla
volta e, se la campagna non e' finita, rilancia il controller successivo con
dipendenza Slurm. Serve a non intasare `squeue` con decine di array gia'
sottomessi.

`submit_bananas_cifar_resume_chain_slurm.sh`

Approccio precedente basato su catene di array con dipendenze Slurm. Funziona,
ma e' meno comodo del controller CPU per campagne lunghe.

`submit_bananas_cifar_requeue_array_slurm.sh`

Approccio sperimentale basato su requeue. E' stato abbandonato perche' meno
trasparente e meno controllabile.

`submit_bananas_cifar_array_slurm.sh`

Script array diretto per BANANAS. Utile per prove semplici o smoke test, ma non
e' lo script consigliato per la campagna completa.

`submit_bananas_cifar_slurm.sh`

Primo script semplice per sottomettere run BANANAS. Conservato come supporto,
ma superato dagli script rolling/controller.

`submit_flexibo_cifar_rolling_array_slurm.sh`

Script worker/array attuale per FlexiBO. Gestisce resume, chunk, partizione GPU
e variabili CUDA/XLA necessarie a evitare l'errore `libdevice.10.bc`.

`submit_flexibo_cifar_cpu_controller_slurm.sh`

Controller leggero su partizione CPU per FlexiBO, analogo a quello BANANAS.
Sottomette un array GPU alla volta e rilancia se ci sono ancora chunk da fare.

### Ambiente e Artefatti

`READMEaltrituner.md`

Documento operativo del branch. Contiene la spiegazione delle implementazioni,
le scelte metodologiche, i comandi Slurm, la lettura dei risultati e questa
mappa dei file aggiunti/modificati.

`.gitignore`

E' stato esteso per ignorare file locali e artefatti pesanti, tra cui
`.codex/`, `results_BANANAS*/`, `slurm_logs*/`, `slurm-*.out` e file modello.

`environment_bananas.yml`

Ambiente Conda usato per BANANAS e FlexiBO. Include TensorFlow 2.15, pacchetti
CUDA/cuDNN installati via pip, `scikit-optimize`, `problog`, `datasets`,
`pandas`, `PyYAML` e le altre dipendenze necessarie ai runner.

`todo.txt`

File di appunti di lavoro del branch. Non e' parte della pipeline eseguibile,
ma tiene traccia delle domande operative sui tuner e sugli esperimenti.

`othertunerdependencies/bananas/naszilla/LICENSE`

Licenza del codice NASzilla vendorizzato.

`othertunerdependencies/flexibo/FlexiBO/LICENSE`

Licenza del codice FlexiBO vendorizzato.

`results_BANANAS_naszilla_controller_cluster/`

Risultati BANANAS scaricati dal cluster. Sono artefatti sperimentali, non codice
sorgente. Possono essere aggiornati con `scarica.sh`.

`results_FLEXIBO_controller_cluster/`

Risultati FlexiBO scaricati dal cluster. Sono artefatti sperimentali, non codice
sorgente. Possono essere aggiornati con `scarica.sh`.

`slurm_logs_cluster/`

Log Slurm scaricati dal cluster. Servono per diagnosticare errori, tempi di run
e stato delle campagne.

`tuner_results_summary.csv`

CSV derivato dai risultati scaricati. Contiene una riga per run.

`tuner_results_aggregates.csv`

CSV derivato dai risultati scaricati. Contiene medie e statistiche aggregate per
tuner, dataset e seed.

`tuner_results_paper_table.csv`

CSV derivato dai risultati scaricati. Contiene una tabella compatta in stile
paper con `Best Score`, `Accuracy`, `MFLOPs` e `N. Iteration` aggregati per
dataset e tuner.

### Modifiche Funzionali Principali

Le modifiche del branch introducono:

- confronto di BANANAS e FlexiBO sullo stesso spazio RS del progetto;
- adapter RS verso BANANAS/NASzilla;
- adapter RS multi-objective verso FlexiBO con accuracy e FLOPs;
- vendoring minimale delle parti necessarie delle repository originali;
- resume/chunking per evitare job GPU troppo lunghi;
- controller CPU leggeri per sottomettere pochi job array alla volta;
- gestione CUDA/XLA negli script Slurm per evitare errori `libdevice.10.bc`;
- script per scaricare risultati/log dal cluster;
- script per esportare risultati e medie in CSV.

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

Questa directory contiene un vendoring minimale dei file NASzilla originali
necessari al runner. Non e' un submodule: i file sono versionati direttamente
in questa repository per semplificare pull e uso sul cluster.

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
5. il neural predictor e le acquisition function arrivano dal codice NASzilla vendorizzato;
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

## FlexiBO

File principale:

```text
flexibo_runner.py
```

Dipendenza originale:

```text
othertunerdependencies/flexibo/FlexiBO
```

Questa directory contiene un vendoring minimale dei file FlexiBO originali
necessari al runner. Non e' un submodule: i file sono versionati direttamente
in questa repository per semplificare pull e uso sul cluster.

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

## Slurm

Dopo un pull sul cluster non servono comandi per submodule. Le dipendenze
esterne minime sono vendorizzate nella repository, quindi basta:

```bash
git pull
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

## CSV Dei Risultati Scaricati

Dopo aver scaricato risultati e log dal cluster con:

```bash
./scarica.sh remote
```

oppure, dalla LAN:

```bash
./scarica.sh lan
```

si puo' generare un CSV riassuntivo locale con:

```bash
./export_tuner_results_csv.py
```

I file prodotti di default sono:

```text
tuner_results_summary.csv
tuner_results_aggregates.csv
tuner_results_paper_table.csv
```

`tuner_results_summary.csv` contiene una riga per ogni run, cioe' per ogni combinazione:

```text
tuner, dataset, seed
```

`tuner_results_aggregates.csv` contiene invece medie e statistiche aggregate.

`tuner_results_paper_table.csv` contiene la tabella gia' aggregata in stile
paper, con una riga per:

```text
dataset, strategy
```

e colonne:

```text
Best Score
Accuracy
MFLOPs
N. Iteration
```

I valori sono formattati come:

```text
media (+- deviazione standard)
```

Per scegliere altri nomi:

```bash
./export_tuner_results_csv.py \
  --output risultati_tuner.csv \
  --aggregates-output risultati_tuner_aggregati.csv \
  --paper-table-output risultati_tuner_tabella_paper.csv
```

Lo script legge:

```text
results_BANANAS_naszilla_controller_cluster/
results_FLEXIBO_controller_cluster/
```

cioe' le directory locali create da `scarica.sh`.

### Significato Delle Colonne

Le colonne principali sono:

```text
tuner
dataset
seed
progress_evals
events
best_score
best_accuracy
```

`progress_evals` indica quante valutazioni con accuracy/training sono state completate.

Per BANANAS:

```text
progress_evals = events
```

perche' ogni evento corrisponde a una rete valutata.

Per FlexiBO:

```text
events >= progress_evals
```

perche' FlexiBO registra anche eventi FLOPs-only, cioe' valutazioni economiche del secondo obiettivo senza training completo. Per confrontare il budget di training/GPU, usare `progress_evals`, non `events`.

### Metriche Principali

Il CSV espone due metriche principali:

```text
best_score
best_accuracy
```

`best_score` e' il migliore valore della funzione obiettivo ottimizzata dal tuner. Piu' basso e' meglio.

Con `flops_module`, lo score non e' solo accuracy: incorpora il tradeoff/controllo legato a FLOPs e parametri. Quindi questa e' la metrica corretta quando si vuole valutare il tuner secondo l'obiettivo del progetto.

`best_accuracy` e' la massima accuracy pura osservata in quella run. E' utile per capire quanto bene il tuner riesce a spingere la performance predittiva, ignorando il fatto che il modello possa essere piu' costoso.

In pratica:

```text
best_score    -> migliore modello secondo l'obiettivo ottimizzato
best_accuracy -> migliore accuracy pura trovata
```

Le colonne `best_score_event`, `best_score_accuracy`, `best_score_flops`,
`best_score_params`, `best_accuracy_event`, `best_accuracy_flops` e
`best_accuracy_params` sono dettagli di supporto: indicano dove e con quale
costo sono stati ottenuti i due valori principali.

### CSV Stile Paper

Il file:

```text
tuner_results_paper_table.csv
```

e' quello piu' simile alle tabelle del paper.

Esempio di colonne:

```text
dataset
strategy
runs
Best Score
Accuracy
MFLOPs
N. Iteration
```

Il significato e':

- `strategy`: tuner usato, ad esempio `BANANAS` o `FlexiBO`;
- `Best Score`: media e deviazione standard del miglior score trovato;
- `Accuracy`: test accuracy del modello che ha ottenuto il miglior score;
- `MFLOPs`: FLOPs, in milioni, del modello che ha ottenuto il miglior score;
- `N. Iteration`: iterazione/evento in cui e' stata identificata la migliore architettura valida;
- `runs`: numero di seed/run aggregate.

Quindi `Accuracy`, `MFLOPs` e `N. Iteration` descrivono lo stesso modello di
`Best Score`. Questo e' diverso da `best_accuracy` nel CSV dettagliato, che
invece indica la massima accuracy pura osservata nella run, anche se ottenuta da
un modello diverso.

Per BANANAS un evento corrisponde a una valutazione completa. Per FlexiBO un
evento puo' essere anche una valutazione FLOPs-only, quindi `N. Iteration`
segue la logica della caption del paper: numero di iterazioni del tuner
necessarie a identificare la migliore architettura valida.

Oltre alle colonne formattate, il CSV contiene anche le colonne numeriche:

```text
best_score_rank
accuracy_rank
mflops_rank
n_iteration_rank
best_score_mean
best_score_std
accuracy_pct_mean
accuracy_pct_std
mflops_mean
mflops_std
n_iteration_mean
n_iteration_std
```

Le colonne `*_rank` permettono di applicare lo stile della caption del paper:
rank `1` in grassetto, rank `2` in corsivo. Per `Best Score`, `MFLOPs` e
`N. Iteration` il valore minore e' migliore; per `Accuracy` il valore maggiore
e' migliore.

Le altre colonne numeriche sono piu' comode per plotting o ulteriori
elaborazioni.

### CSV Aggregati

Il file:

```text
tuner_results_aggregates.csv
```

contiene medie gia' pronte per leggere i risultati senza dover aggregare a mano.

La colonna `group_type` indica il tipo di aggregazione:

```text
tuner
tuner_dataset_mean_over_seeds
tuner_seed_mean_over_datasets
```

`tuner` aggrega tutte le run di un tuner, quindi fa la media su dataset e seed.

`tuner_dataset_mean_over_seeds` aggrega per:

```text
tuner, dataset
```

quindi e' la media su tutti i seed dello stesso dataset. Questa e' di solito la riga piu' utile per confrontare BANANAS e FlexiBO su CIFAR10 o CIFAR100.

`tuner_seed_mean_over_datasets` aggrega per:

```text
tuner, seed
```

quindi e' la media dello stesso seed sui dataset disponibili. Serve soprattutto per controllare se un seed e' sistematicamente piu' favorevole o sfavorevole.

Le colonne aggregate principali sono:

```text
mean_progress_evals
mean_best_score
std_best_score
min_best_score
max_best_score
mean_best_score_event
std_best_score_event
mean_best_score_accuracy
std_best_score_accuracy
mean_best_accuracy
std_best_accuracy
min_best_accuracy
max_best_accuracy
```

Sono presenti anche medie di supporto per FLOPs, parametri ed eventi penalizzati:

```text
mean_best_score_flops
mean_best_score_params
mean_best_accuracy_flops
mean_best_accuracy_params
mean_invalid_or_penalty_events
```

### Medie Snapshot Locale Per Dataset

Le seguenti medie sono state calcolate dallo snapshot locale corrente con:

```bash
./export_tuner_results_csv.py
```

Sono le righe `tuner_dataset_mean_over_seeds` di `tuner_results_aggregates.csv`.

| tuner | dataset | runs | mean progress evals | mean best score | mean best accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| BANANAS | cifar10 | 5 | 150.0 | -0.7872 | 0.7635 |
| BANANAS | cifar100 | 5 | 143.8 | -0.5310 | 0.4017 |
| FlexiBO | cifar10 | 5 | 230.6 | -0.7924 | 0.7562 |
| FlexiBO | cifar100 | 5 | 220.0 | -0.5156 | 0.4052 |

Queste medie sono uno snapshot intermedio: non tutte le run hanno lo stesso numero di valutazioni e nessuna e' ancora arrivata a `1000/1000`.

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
