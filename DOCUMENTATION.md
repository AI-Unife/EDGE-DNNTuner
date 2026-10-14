# Symbolic DNN-Tuner — Documentazione Tecnica

> **Autori originali:** Michele Fraccaroli, Evelina Lamma, Fabrizio Riguzzi  
> **Repository:** [micheleFraccaroli/Symbolic_DNN-Tuner](https://github.com/micheleFraccaroli/Symbolic_DNN-Tuner)  
> **Branch attivo:** `main`

---

## Indice

1. [Panoramica del Progetto](#1-panoramica-del-progetto)
2. [Architettura del Sistema](#2-architettura-del-sistema)
3. [Struttura delle Directory](#3-struttura-delle-directory)
4. [Configurazione (`exp_config.py`)](#4-configurazione-exp_configpy)
5. [Entry Point Principale (`symbolic_tuner.py`)](#5-entry-point-principale-symbolic_tunerpy)
6. [Componenti Core (`components/`)](#6-componenti-core-components)
   - 6.1 [Controller](#61-controller)
   - 6.2 [Search Space](#62-search-space)
   - 6.3 [Objective Function e Constraints](#63-objective-function-e-constraints)
   - 6.4 [Neural Network (Abstract)](#64-neural-network-abstract)
   - 6.5 [Model Interface e Backend Interface](#65-model-interface-e-backend-interface)
   - 6.6 [Neural-Symbolic Bridge](#66-neural-symbolic-bridge)
   - 6.7 [Tuning Rules Symbolic](#67-tuning-rules-symbolic)
   - 6.8 [LFI Integration](#68-lfi-integration)
   - 6.9 [Dataset](#69-dataset)
   - 6.10 [Improvement Checker](#610-improvement-checker)
   - 6.11 [Integral](#611-integral)
   - 6.12 [Random Search](#612-random-search)
   - 6.13 [Storing Experience](#613-storing-experience)
   - 6.14 [Bayesian Optimization (`bo/`)](#614-bayesian-optimization-bo)
7. [Moduli di Loss (`modules/`)](#7-moduli-di-loss-modules)
   - 7.1 [Module Manager](#71-module-manager)
   - 7.2 [Common Interface](#72-common-interface)
   - 7.3 [FLOPs Module](#73-flops-module)
   - 7.4 [Hardware Module](#74-hardware-module)
8. [Implementazioni Framework-Specific](#8-implementazioni-framework-specific)
   - 8.1 [PyTorch](#81-pytorch)
   - 8.2 [TensorFlow](#82-tensorflow)
9. [Ragionamento Simbolico (`symbolic_base/`)](#9-ragionamento-simbolico-symbolic_base)
10. [FLOPs & Profilazione Hardware](#10-flops--profilazione-hardware)
    - 10.1 [Calculator FLOPs (`flops/`)](#101-calculator-flops-flops)
    - 10.2 [Profiler NVDLA (`nvdla/`)](#102-profiler-nvdla-nvdla)
11. [Quantizzazione (`quantizer/`)](#11-quantizzazione-quantizer)
12. [Script di Utilità](#12-script-di-utilità)
13. [Esecuzione su Cluster HPC (SLURM)](#13-esecuzione-su-cluster-hpc-slurm)
14. [Struttura dei Log di Algoritmo](#14-struttura-dei-log-di-algoritmo)
15. [Debug Mode](#15-debug-mode)
16. [Dipendenze](#16-dipendenze)
17. [Riferimenti Bibliografici](#17-riferimenti-bibliografici)

---

## 1. Panoramica del Progetto

**Symbolic DNN-Tuner** è un sistema di ottimizzazione automatica degli iperparametri (AutoML) per reti neurali profonde che combina:

- **Bayesian Optimization (BO)** o **Random Search (RS)** per il campionamento degli iperparametri;
- **Ragionamento simbolico** tramite ProbLog per la diagnosi intelligente dei problemi di training e la proposta di azioni correttive;
- **Learning From Interpretations (LFI)** per l'adattamento probabilistico dei pesi delle regole simboliche sulla base dell'esperienza accumulata;
- **Funzioni di loss modulari** per l'ottimizzazione multi-obiettivo (accuracy, FLOPs, latenza hardware).

Il sistema supporta due backend deep learning (TensorFlow e PyTorch), gestisce dataset standard (CIFAR-10/100, MNIST, TinyImageNet) e include profilazione hardware su NVDLA e quantizzazione opzionale.

---

## 2. Architettura del Sistema

```
┌─────────────────────────────────────────────────────────┐
│                   symbolic_tuner.py                      │
│              (Orchestratore principale)                   │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────▼────────────────┐
         │     exp_config.py              │
         │  (Configurazione YAML)         │
         └───────────────┬────────────────┘
                         │
         ┌───────────────▼──────────────────────────────┐
         │          LOOP DI OTTIMIZZAZIONE               │
         │     (max_iter iterazioni o convergenza)       │
         └───────────────┬──────────────────────────────┘
                         │
    ┌────────────────────┼────────────────────────┐
    ▼                    ▼                        ▼
┌────────┐     ┌──────────────┐         ┌──────────────┐
│BO / RS │     │  controller  │         │  modules/    │
│(skopt) │     │  (training,  │         │  loss/       │
│        │     │   diagnosis, │         │  (flops,     │
│        │     │   tuning)    │         │   hardware)  │
└───┬────┘     └──────┬───────┘         └──────┬───────┘
    │                 │                        │
    │     ┌───────────▼─────────────┐          │
    │     │ NeuralSymbolicBridge    │◄─────────┘
    │     │ (ProbLog reasoning)     │
    │     └───────────┬─────────────┘
    │                 │
    │     ┌───────────▼─────────────┐
    │     │ tuning_rules_symbolic   │
    │     │ (Mutazione dello spazio)│
    │     └───────────┬─────────────┘
    │                 │
    ▼                 ▼
┌──────────────────────────────────────┐
│  Backend Framework (TF / PyTorch)    │
│  - NeuralNetwork (build, train)      │
│  - ModuleBackend (FLOPs, params)     │
│  - Model (architettura dinamica)     │
└──────────────────────────────────────┘
```

### Flusso di Esecuzione (Singola Iterazione)

1. **Campionamento:** BO/RS propone un vettore di iperparametri dallo spazio di ricerca corrente.
2. **Training:** Il `controller` costruisce e addestra la rete, verificando i vincoli su FLOPs/latenza.
3. **Diagnosi simbolica:** Il `NeuralSymbolicBridge` traduce le metriche numeriche in fatti Prolog, esegue l'inferenza probabilistica e identifica problemi (overfitting, underfitting, ecc.).
4. **Tuning:** Le `tuning_rules_symbolic` applicano le azioni consigliate (aggiungere/rimuovere layer, modificare i bound degli iperparametri).
5. **Aggiornamento spazio:** Lo spazio di ricerca viene espanso/restretto in base alla diagnosi.
6. **LFI:** I pesi delle regole simboliche vengono aggiornati in base al miglioramento osservato.

---

## 3. Struttura delle Directory

```
Symbolic_DNN-Tuner/
├── symbolic_tuner.py          # Entry point principale
├── exp_config.py              # Sistema di configurazione (YAML + dataclass)
├── analyze_results.py         # Analisi post-esperimento dei risultati
├── download_hf_dataset.py     # Download dataset da HuggingFace
├── test_quantizer.py          # Test del quantizzatore
├── requirements.txt           # Dipendenze Python
│
├── components/                # Logica core del sistema
│   ├── controller.py          # Orchestratore training/diagnosis/tuning
│   ├── search_space.py        # Gestione spazio di ricerca (skopt.Space)
│   ├── neural_network.py      # ABC per reti neurali
│   ├── model_interface.py     # Strutture dati per layer e modelli
│   ├── backend_interface.py   # ABC per backend framework
│   ├── neural_sym_bridge.py   # Ponte numerico ↔ simbolico
│   ├── objFunction.py         # Wrapper funzione obiettivo
│   ├── constraints.py         # Vincoli sullo spazio campionato
│   ├── dataset.py             # Caricamento dataset
│   ├── improvement_checker.py # Tracciamento miglioramenti
│   ├── integral.py            # Calcolo features integrali delle curve di loss
│   ├── lfi_integration.py     # Interfaccia ProbLog LFI
│   ├── random_search.py       # Ottimizzatore Random Search
│   ├── storing_experience.py  # Persistenza esperienza (SQLite)
│   ├── tuning_rules_symbolic.py # Regole di mutazione architettura/HP
│   ├── colors.py              # Costanti colore ANSI per terminale
│   ├── optimizers/            # Strategie `opt`: registry + sampler
│   │   ├── __init__.py        # Registry STRATEGIES e get_strategy()
│   │   ├── base.py            # Interfaccia Sampler + dataclass OptStrategy
│   │   ├── bo.py              # BOSampler (gp_minimize custom o skopt)
│   │   └── random_search.py   # RSSampler (wrapper di components/random_search.py)
│   └── bo/                    # Bayesian Optimization personalizzata
│       ├── base.py            # Funzione base_minimize
│       ├── gp.py              # gp_minimize (GP-based BO)
│       └── optimizer.py       # Optimizer stateful
│
├── modules/                   # Moduli di loss pluggabili
│   ├── module.py              # Manager dei moduli
│   ├── common_interface.py    # ABC per moduli di loss
│   └── loss/                  # Moduli concreti
│       ├── flops_module.py    # Modulo FLOPs/parametri
│       ├── flops_module.pl    # Regole Prolog per FLOPs
│       ├── hardware_module.py # Modulo latenza hardware NVDLA
│       └── hardware_module.pl # Regole Prolog per hardware
│
├── pytorch_implementation/    # Implementazione PyTorch
│   ├── model.py               # Builder modello (ConvBlock, TorchModel)
│   ├── module_backend.py      # Backend FLOPs/params per PyTorch
│   └── neural_network.py      # Training loop PyTorch
│
├── tensorflow_implementation/ # Implementazione TensorFlow
│   ├── model.py               # Builder modello Keras
│   ├── module_backend.py      # Backend FLOPs/params per TensorFlow
│   └── neural_network.py      # Training loop TensorFlow
│
├── symbolic_base/             # File base di ragionamento simbolico
│   ├── symbolic_analysis.pl   # Regole Prolog per diagnosi problemi
│   ├── sym_prob_base.pl       # Regole probabilistiche per azioni
│   └── lfi.pl                 # Template per Learning From Interpretations
│
├── flops/                     # Calcolo FLOPs
│   ├── flops_calculator.py    # Profiler FLOPs TensorFlow
│   └── node_manager.py        # Conversione grafo profiler in dizionario
│
├── nvdla/                     # Profilazione hardware NVDLA
│   ├── profiler.py            # Profiler latenza NVDLA
│   ├── profilerEMBER.py       # Profiler integrato con EMBER
│   ├── models/                # Configurazioni hardware
│   ├── specs/                 # Specifiche acceleratore
│   └── wrapper/               # Wrapper per interfaccia C++
│
└──quantizer/                 # Quantizzazione modelli
    ├── quantizer_interface.py # ABC per quantizzatori
    ├── quantizer_POTQ.py      # Post-Training Quantization
    └── binary_quantizer.py    # Quantizzazione binaria dei pesi
```

---

## 4. Configurazione (`exp_config.py`)

Il sistema di configurazione utilizza un `dataclass` (`ConfigSchema`) con valori di default e un file YAML persistente.

### Parametri Principali

| Parametro       | Tipo      | Default             | Descrizione                                                   |
|-----------------|-----------|---------------------|---------------------------------------------------------------|
| `backend`       | `str`     | `"tf"`              | Framework: `"tf"` (TensorFlow) o `"torch"` (PyTorch)         |
| `eval`          | `int`     | `300`               | Numero massimo di valutazioni                                 |
| `early_stop`    | `int`     | `30`                | Patience per early stopping                                   |
| `epochs`        | `int`     | `2`                 | Epoche di training per valutazione                            |
| `mod_list`      | `list`    | `[]`                | Moduli attivi (es. `["flops_module", "hardware_module"]`)     |
| `dataset`       | `str`     | `"light"`           | Dataset: `cifar10`, `cifar100`, `mnist`, `tinyimagenet`, ecc. |
| `name`          | `str`     | `"experiment"`      | Nome esperimento (determina la directory di output)           |
| `seed`          | `int`     | `42`                | Seed per riproducibilità                                      |
| `quantization`  | `bool`    | `False`             | Abilitare la quantizzazione post-training                     |
| `verbose`       | `int`     | `2`                 | Livello di verbosità (0: silenzioso, 1: spazio, 2: spazio+modello) |
| `opt`           | `str`     | `"filtered"`        | Strategia di ottimizzazione                                   |
| `w_flops`       | `float`   | `0.33`              | Peso della loss FLOPs                                         |
| `w_HW`          | `float`   | `0.33`              | Peso della loss hardware                                      |
| `lacc`          | `float`   | `0.10`              | Soglia per underfitting: se `1 - acc > lacc`                  |
| `flops_th`      | `int`     | `150_000_000`       | Soglia massima FLOPs                                          |
| `nparams_th`    | `int`     | `2_500_000`         | Soglia massima parametri                                      |
| `hf_cache`      | `str`     | `None`              | Directory cache HuggingFace                                   |

### Strategie di Ottimizzazione (`opt`)

| Strategia    | Descrizione                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `standard`   | BO pura senza regole simboliche; lo spazio di ricerca è fisso               |
| `filtered`   | BO con constraint filtering + ragionamento simbolico + mutazione spazio     |
| `basic`      | BO con regole simboliche ma senza espansione dello spazio base              |
| `RS`         | Random Search pura senza regole simboliche                                  |
| `RS_ruled`   | Random Search con regole simboliche e mutazione dello spazio                |

Ogni strategia è dichiarata in `components/optimizers/__init__.py` (`STRATEGIES`) come
`OptStrategy`: un sampler (`custom_bo`, `skopt_bo`, `random_search`), una politica dello
spazio (`static`, `expand`, `replace`) e se applicare il filtro dei vincoli.
`symbolic_tuner.py`, `search_space.py` ed `exp_config.py` leggono solo la strategia: una nuova
combinazione è una riga nel registry, un nuovo algoritmo di ricerca è una nuova sottoclasse
di `Sampler`.

### Ciclo di Vita della Configurazione

```python
# 1. Creazione del file YAML con override da CLI
cfg_path = create_config_file(exp_dir, overrides=args.__dict__)

# 2. Impostazione come config attiva (variabile d'ambiente)
set_active_config(cfg_path)

# 3. Caricamento (con caching thread-safe)
cfg = load_cfg(force=True)

# 4. Accesso ai parametri tramite dot-notation
print(cfg.backend)   # "tf"
print(cfg.epochs)    # 2
```

---

## 5. Entry Point Principale (`symbolic_tuner.py`)

Lo script `symbolic_tuner.py` è il punto di ingresso dell'intero sistema. La sua esecuzione segue questi passi:

### 5.1 Parsing degli Argomenti

`parse_args()` definisce i flag CLI che corrispondono ai parametri di `ConfigSchema`. Questi sovrascrivono i default nel file YAML.

### 5.2 Setup dell'Esperimento

```python
# Crea la directory e il config.yaml
cfg_path = create_config_file(exp_dir, overrides=args.__dict__)
set_active_config(cfg_path)
cfg = load_cfg(force=True)

# Import dinamico del backend
if cfg.backend == "tf":
    from tensorflow_implementation import module_backend, neural_network
elif cfg.backend == "torch":
    from pytorch_implementation import module_backend, neural_network

# Creazione directory esperimento e copia file simbolici
create_experiment_folders()
copy_symbolic_files()
```

### 5.3 Caricamento Dataset

Il `TunerDataset` carica il dataset specificato da `cfg.dataset` e lo suddivide in train/validation/test.

### 5.4 Inizializzazione Controller

Il `controller` è il cuore del sistema. Riceve la classe NN, il backend dei moduli e il dataset.

### 5.5 Loop di Ottimizzazione (`run_optimization`)

```python
while ctrl.iter <= max_iter and not ctrl.convergence:
    # 1. Campiona iperparametri con BO o RS
    res = gp_minimize(...) | random_search(...)

    # 2. Aggiorna cronologia x0, y0
    x0, y0 = res.x_iters, res.func_vals

    # 3. Diagnosi simbolica e mutazione spazio
    if cfg.opt in ["filtered", "RS_ruled", "basic"]:
        next_space = ctrl.diagnosis(const_space)
    else:
        next_space = copy.deepcopy(base_space)

    # 4. Espansione spazio se necessario
    base_space = first_ss.expand_space(base_space, next_space)

    # 5. Reset optimizer se le dimensioni cambiano
    if len(next_space.dimensions) != len(const_space.dimensions):
        x0, y0 = [], []

    const_space = copy.deepcopy(next_space)
```

---

## 6. Componenti Core (`components/`)

### 6.1 Controller

**File:** `components/controller.py`

Il `controller` è l'**orchestratore centrale** del sistema. Gestisce tre fasi principali per ogni iterazione:

- **`training(params)`**: Costruisce la rete neurale con gli iperparametri proposti, verifica i vincoli (FLOPs, latenza), esegue il training e restituisce lo score.
- **`diagnosis(const_space)`**: Calcola feature integrali dalla curva di loss, invoca il ragionamento simbolico ProbLog, identifica i problemi e propone azioni correttive.
- **`tuning(actions, space)`**: Applica le azioni di tuning (aggiunta/rimozione layer, modifica bound) allo spazio di ricerca.

Tiene traccia dello stato globale: iterazione corrente, miglior score, contatori dei layer, flag di convergenza.

### 6.2 Search Space

**File:** `components/search_space.py`

Classe `search_space` che gestisce lo spazio di ricerca basato su `skopt.Space`. Operazioni principali:

- **`search_sp(max_block, max_dense)`**: Genera lo spazio di ricerca iniziale con le dimensioni per learning rate, batch size, numero di blocchi convoluzionali, neuroni FC, dropout, ecc.
- **`add_params(space, params)`**: Aggiunge nuove dimensioni (es. quando viene aggiunto un nuovo blocco convoluzionale).
- **`remove_params(space, names)`**: Rimuove dimensioni dallo spazio.
- **`expand_space(base, new)`**: Unisce lo spazio base con quello nuovo, mantenendo le dimensioni esistenti e aggiungendo quelle nuove.

### 6.3 Objective Function e Constraints

**File:** `components/objFunction.py` e `components/constraints.py`

- **`ObjectiveWrapper`**: Adatta l'interfaccia tra l'ottimizzatore (che passa una lista di valori) e il `controller.training()` (che riceve un dizionario di parametri). Converte la lista posizionale in un dizionario nome→valore basato sulle dimensioni dello spazio corrente.

- **`ConstraintsWrapper`**: Implementa `apply_constraints()` che viene passata a `gp_minimize` come `space_constraint` nella modalità `filtered`. Filtra i campioni invalidi durante la fase di acquisizione del BO.

### 6.4 Neural Network (Abstract)

**File:** `components/neural_network.py`

Classe base astratta (`ABC`) che definisce l'interfaccia framework-agnostica per le reti neurali:

| Metodo Astratto       | Descrizione                                        |
|-----------------------|----------------------------------------------------|
| `build_network(params)` | Costruisce il modello dalla configurazione HP      |
| `training(params)`    | Esegue il ciclo di training                         |
| `save_model(path)`    | Salva il modello su disco                          |
| `eval_model(data)`    | Valuta il modello sui dati di test                 |

Le implementazioni concrete risiedono in `pytorch_implementation/` e `tensorflow_implementation/`.

### 6.5 Model Interface e Backend Interface

**File:** `components/model_interface.py` e `components/backend_interface.py`

- **`model_interface.py`** definisce:
  - `LayerSpec`: struttura dati per specificare un layer (tipo, filtri, kernel, ecc.)
  - `TunerModel`: classe base per il modello, con metodi di costruzione da un dizionario di parametri
  - `LayerTypes`: enumerazione dei tipi di layer supportati
  - `Params`: classe per gestire i parametri in modo tipizzato

- **`backend_interface.py`** definisce l'ABC `BackendInterface` per operazioni che dipendono dal framework:
  - `get_flops(model)`: calcola i FLOPs
  - `get_params(model)`: conta i parametri
  - `get_layer_info(model)`: estrae informazioni sui layer

### 6.6 Neural-Symbolic Bridge

**File:** `components/neural_sym_bridge.py`

Il `NeuralSymbolicBridge` è il **componente chiave** che collega il mondo numerico (metriche di training) al ragionamento simbolico (ProbLog):

1. **`build_symbolic_model()`**: Assembla un programma Prolog combinando:
   - Fatti numerici (accuracy, loss, FLOPs, ecc.)
   - Regole probabilistiche dal file `sym_prob_base.pl`
   - Regole di analisi da `symbolic_analysis.pl`
   - Regole dei moduli attivi (es. `flops_module.pl`)

2. **`symbolic_reasoning()`**: Esegue l'inferenza ProbLog, interrogando gli atomi `action(_, _)` e restituendo le probabilità.

3. **`edit_probs()`**: Aggiorna le probabilità delle regole in base ai pesi appresi dall'LFI (se c'è stato miglioramento).

### 6.7 Tuning Rules Symbolic

**File:** `components/tuning_rules_symbolic.py`

Contiene oltre **20 metodi** di mutazione architettura/iperparametri. Ogni metodo è un'azione che il ragionamento simbolico può raccomandare:

| Categoria       | Azioni                                                            |
|-----------------|-------------------------------------------------------------------|
| **Architettura** | `new_conv_block()`, `remove_conv_block()`, `new_fc_layer()`, `remove_fc_layer()` |
| **Regolarizzazione** | `inc_dropout()`, `decr_dropout()`, `reg_l1()`, `reg_l2()`  |
| **Learning Rate** | `decr_lr()`, `inc_lr()`                                        |
| **Batch Size**  | `inc_batch()`, `decr_batch()`                                     |
| **Data Augment** | `data_augmentation()`                                            |
| **Filtri**       | `inc_filters()`, `decr_filters()`                               |

Ogni azione modifica lo spazio di ricerca tramite `search_space.add_params()` o `search_space.remove_params()`.

### 6.8 LFI Integration

**File:** `components/lfi_integration.py`

Wrapper per il **Learning From Interpretations** di ProbLog. Impara i pesi delle regole probabilistiche dall'esperienza accumulata:

1. **`create_evidence(action, problem, improved)`**: Crea tuple di evidenza (es. *"l'azione `inc_dropout` per `overfitting` ha portato a un miglioramento"*).
2. **`learning()`**: Carica il programma LFI da `lfi.pl`, esegue `run_lfi()` con l'esperienza accumulata, ed estrae i pesi aggiornati.
3. I pesi appresi vengono passati al `NeuralSymbolicBridge` per aggiornare le probabilità delle regole.

### 6.9 Dataset

**File:** `components/dataset.py`

- **`TunerDataset`**: Caricatore unificato per i dataset standard. Ogni metodo `load_*()` restituisce `(x_train, y_train, x_val, y_val, x_test, y_test)` e i metadati (`num_classes`, `input_shape`).

| Dataset         | Metodo                 | Input Shape        |
|-----------------|------------------------|--------------------|
| CIFAR-10        | `load_cifar_10()`      | (32, 32, 3)        |
| CIFAR-100       | `load_cifar_100()`     | (32, 32, 3)        |
| MNIST           | `load_mnist()`         | (28, 28, 1)        |
| TinyImageNet    | `load_tiny_imagenet()`  | (64, 64, 3)        |
| Light (subset)  | `load_light_cifar()`   | (32, 32, 3)        |

### 6.10 Improvement Checker

**File:** `components/improvement_checker.py`

Classe `ImprovementChecker` che confronta lo score corrente con la cronologia per determinare se c'è stato un miglioramento. Il risultato booleano viene usato come evidenza per l'LFI.

### 6.11 Integral

**File:** `components/integral.py`

Funzione `integrals()` che calcola le feature integrali dalle curve di loss di validazione:
- **Area sotto la curva (AUC)** della loss
- **Pendenza (slope)** della loss

Queste metriche vengono tradotte in fatti Prolog (es. `slow_start`, `high_loss`) per il ragionamento simbolico.

### 6.12 Random Search

**File:** `components/random_search.py`

Classe `RandomSearch` come alternativa alla Bayesian Optimization. Campiona uniformemente dallo spazio di ricerca. Utilizzata quando `cfg.opt` contiene `"RS"`.

### 6.13 Storing Experience

**File:** `components/storing_experience.py`

Classe `StoringExperience` che implementa un database SQLite per la persistenza dell'esperienza di tuning:
- `insert_ranking(score, params)`: Salva il risultato di ogni iterazione
- `insert_evidence(action, problem, improved)`: Salva le evidenze per LFI
- `get()`: Recupera le evidenze storiche

### 6.14 Bayesian Optimization (`bo/`)

**File:** `components/bo/base.py`, `components/bo/gp.py`, `components/bo/optimizer.py`

Implementazione personalizzata della Bayesian Optimization basata su `scikit-optimize`:

- **`gp.py`**: Espone `gp_minimize()` con supporto per `space_constraint` (usata nella modalità `filtered`)
- **`base.py`**: Funzione `base_minimize()` che gestisce il loop interno BO
- **`optimizer.py`**: Classe `Optimizer` stateful per utilizzo avanzato

---

## 7. Moduli di Loss (`modules/`)

### 7.1 Module Manager

**File:** `modules/module.py`

La classe `module` gestisce il caricamento dinamico e l'orchestrazione dei moduli di loss:

- **`load_modules(mod_list)`**: Carica dinamicamente i moduli specificati in `cfg.mod_list`
- **`get_rules()`**: Raccoglie le regole Prolog da tutti i moduli attivi
- **`state()`**: Aggiorna lo stato di tutti i moduli con le metriche correnti
- **`values()`**: Restituisce i fatti Prolog da tutti i moduli
- **`optimization()`**: Calcola il contributo di loss aggregato di tutti i moduli

### 7.2 Common Interface

**File:** `modules/common_interface.py`

ABC che definisce il contratto per tutti i moduli di loss:

```python
class common_interface(ABC):
    facts: List[str]           # Nomi dei fatti Prolog esportati
    problems: List[str]        # Nomi dei problemi diagnosticabili
    weight: float              # Peso nel contributo alla loss totale

    @abstractmethod
    def update_state(self, *args) -> None: ...
    def obtain_values(self) -> Dict[str, Any]: ...
    def optimiziation_function(self) -> float: ...
    def printing_values(self) -> None: ...
    def log_function(self) -> None: ...
```

### 7.3 FLOPs Module

**File:** `modules/loss/flops_module.py` + `modules/loss/flops_module.pl`

Modulo per la gestione dei vincoli su FLOPs e numero di parametri:

- **Python**: Calcola il gap tra FLOPs/parametri correnti e le soglie, restituisce una penalità scalare
- **Prolog**: Definisce regole probabilistiche che collegano i fatti (es. `flops_over_threshold`) ai problemi (es. `latency`, `model_size`)

### 7.4 Hardware Module

**File:** `modules/loss/hardware_module.py` + `modules/loss/hardware_module.pl`

Modulo per la profilazione della latenza su acceleratore NVDLA:

- **Python**: Chiama il profiler NVDLA per stimare latenza e costi energetici su 3 configurazioni hardware
- **Prolog**: Collega la latenza hardware al problema `out_range`

---

## 8. Implementazioni Framework-Specific

### 8.1 PyTorch

**Directory:** `pytorch_implementation/`

| File                | Descrizione                                                            |
|---------------------|------------------------------------------------------------------------|
| `model.py`          | Definisce `ConvBlock` (Conv→BN→ReLU→Pool→Dropout) e `TorchModel` che estende `nn.Module` e `TunerModel`. L'architettura è costruita dinamicamente dal dizionario di parametri. |
| `module_backend.py` | Implementa `BackendInterface` per PyTorch. Usa `FlopCounterMode` per il conteggio preciso dei FLOPs. |
| `neural_network.py` | Implementa il ciclo di training PyTorch con early stopping, salvataggio del miglior modello e logging. |

### 8.2 TensorFlow

**Directory:** `tensorflow_implementation/`

| File                | Descrizione                                                            |
|---------------------|------------------------------------------------------------------------|
| `model.py`          | Definisce `TFModel` basato su Keras Sequential/Functional API. Supporta connessioni residuali. |
| `module_backend.py` | Implementa `BackendInterface` per TensorFlow, delegando al `flops_calculator`.  |
| `neural_network.py` | Implementa il ciclo di training Keras. |

---

## 9. Ragionamento Simbolico (`symbolic_base/`)

Il sistema utilizza **ProbLog** per il ragionamento probabilistico. I file `.pl` definiscono:

### `symbolic_analysis.pl`

Regole Prolog per la **diagnosi dei problemi** di training a partire dalle metriche:

```prolog
% Esempio: diagnosi di overfitting
problem(overfitting) :- gap_tr_te_acc.
gap_tr_te_acc :- a(A), va(VA), last(A,LTA), last(VA,ScoreA), abs2(LTA,ScoreA,Gap), Gap > 0.1.

% Esempio: diagnosi di slow start
problem(slow_start) :- integral_high, slope_low.
```

### `sym_prob_base.pl`

Regole **probabilistiche** che associano le azioni ai problemi diagnosticati:

```prolog
0.50::action(reg_l2, overfitting) :- problem(overfitting).
0.99::action(data_augmentation, underfitting) :- problem(underfitting).
0.70::action(inc_dropout, overfitting) :- problem(overfitting).
```

Le probabilità (es. `0.50`) sono i pesi iniziali che vengono aggiornati dall'LFI.

### `lfi.pl`

Template per le probabilità **apprendibili** in formato LFI di ProbLog:

```prolog
t(0.50)::action(inc_dropout, overfitting).
t(0.70)::action(decr_lr, slow_convergence).
```

Il prefisso `t(...)` indica che il valore è un parametro apprendibile.

---

## 10. FLOPs & Profilazione Hardware

### 10.1 Calculator FLOPs (`flops/`)

- **`flops_calculator.py`**: Classe `flop_calculator` che utilizza il profiler TensorFlow per calcolare i FLOPs di un modello. Metodo principale: `get_flops(model)`.
- **`node_manager.py`**: Utility `to_dict()` che converte ricorsivamente il grafo del profiler TF in un dizionario operazione → FLOPs.

### 10.2 Profiler NVDLA (`nvdla/`)

- **`profiler.py`**: Classe `nvdla` che modella l'acceleratore NVIDIA NVDLA. Carica le configurazioni hardware da file YAML e calcola i tempi di esecuzione per le operazioni:
  - `getCONVTime()`: Tempo di convoluzione
  - `getSDPTime()`: Tempo SDP (normalization, activation)
  - `getPDPTime()`: Tempo pooling

- **`profilerEMBER.py`**: Versione integrata con il framework EMBER per profilazione su specifiche di acceleratore custom.

Le configurazioni hardware supportate definiscono diverse architetture NVDLA con parametri come MAC array size, bandwidth, clock frequency.

---

## 11. Quantizzazione (`quantizer/`)

Sistema opzionale di quantizzazione post-training, attivabile con `--quantization`:

| File                    | Descrizione                                                      |
|-------------------------|------------------------------------------------------------------|
| `quantizer_interface.py` | ABC per quantizzatori: `quantizer_function()`, `evaluate_quantized_model()`, `save_quantized_model()` |
| `quantizer_POTQ.py`    | Post-Training Quantization (quantizzazione dei pesi e delle attivazioni) |
| `binary_quantizer.py`  | Quantizzazione binaria dei pesi (1 bit)                          |

---

## 12. Script di Utilità

| Script                   | Descrizione                                                                  |
|--------------------------|------------------------------------------------------------------------------|
| `analyze_results.py`    | Analizza i log degli esperimenti completati. Crea CSV per esperimento e un CSV riassuntivo con il miglior risultato per ciascuno. Legge da `algorithm_logs/`, `config.yaml`, e i log dei moduli. |
| `multi_sbatch.py`       | Genera file `params.txt` con tutte le combinazioni di parametri per i job array SLURM. Supporta configurazioni per CIFAR e TinyImageNet. |
| `download_hf_dataset.py` | Scarica dataset da HuggingFace (es. TinyImageNet) nella cache locale. |
| `test_quantizer.py`     | Script di test per il modulo di quantizzazione.                              |

---

## 13. Esecuzione su Cluster HPC (SLURM)

Il progetto include script SLURM per l'esecuzione su cluster HPC:

### Esecuzione Singola (`run.slurm`)

```bash
#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

python symbolic_tuner.py --name test_dataset --dataset tinyimagenet \
    --eval 2 --epochs 3 --backend tf
```

### Job Array — Dataset Immagine (`jonarray.slurm`)

Esegue combinazioni di dataset × ottimizzatore × seed da `params.txt`:

```bash
#SBATCH --array=0-4%4
# Parametri letti da params.txt: DATA,OPT,SEED

python symbolic_tuner.py \
  --name "results_FLOPS/${NAME_EXP}" \
  --dataset "$DATA" --opt "$OPT" --seed "$SEED" \
  --eval 1000 --epochs 100 --mod_list flops_module
```

### Esempio di Utilizzo Locale

```bash
# Esperimento minimo su CIFAR-10
python symbolic_tuner.py \
  --name my_experiment \
  --dataset cifar10 \
  --backend torch \
  --opt filtered \
  --eval 50 \
  --epochs 10 \
  --mod_list flops_module \
  --seed 42

# Analisi dei risultati
python analyze_results.py
```

---

## 14. Struttura dei Log di Algoritmo

La directory **`algorithm_logs/`** contiene un insieme completo di file che registrano il comportamento e i risultati dell'algoritmo di ottimizzazione simbolico durante l'esecuzione di un esperimento. Questi file consentono l'analisi dettagliata del processo di tuning, la ricostruzione della storia di ottimizzazione e l'apprendimento del sistema.

### 14.1 File di Log

#### 14.1.1 `hyper-neural.txt` — Spazio Iperparametrico

**Descrizione**: Contiene un dizionario Python per ogni iterazione di ottimizzazione, registrando i valori degli iperparametri testati.

**Formato**: Una lista Python per riga, dove ogni elemento è un dizionario con chiavi:
- `num_neurons`: numero di neuroni nei layer densi (es. 8, 16, 32, 64)
- `unit_c1`, `unit_c2`: numero di filtri nei blocchi convoluzionali
- `dr_f`: dropout rate nel layer denso (es. 0.1-0.8)
- `learning_rate`: tasso di apprendimento (es. 0.0001-0.01)
- `batch_size`: dimensione del batch (es. 16, 32, 64)
- `optimizer`: tipo di ottimizzatore (SGD, Adam, Adamax, Adadelta, Adagrad, RMSprop)
- `activation`: funzione di attivazione (relu, elu, swish, selu)
- `data_augmentation`: 0 o 1 (booleano)
- `reg_l2`: regolarizzazione L2 (0 o valore > 0)
- `skip_connection`: 0 o 1 (booleano per residual connections)
- `new_fc_X`: numero di layer densi aggiuntivi

**Esempio**:
```python
# Iterazione 1
{'num_neurons': 32, 'unit_c1': 4, 'unit_c2': 8, 'dr_f': 0.5, 'learning_rate': 0.001, 'batch_size': 32, 'optimizer': 'adam', 'activation': 'relu', 'data_augmentation': 0, 'reg_l2': 0.0001, 'skip_connection': 0, 'new_fc_1': 0}

# Iterazione 2
{'num_neurons': 24, 'unit_c1': 3, 'unit_c2': 6, 'dr_f': 0.3, 'learning_rate': 0.0005, 'batch_size': 16, 'optimizer': 'sgd', 'activation': 'elu', 'data_augmentation': 1, 'reg_l2': 0.0, 'skip_connection': 1, 'new_fc_1': 1}
```

#### 14.1.2 `acc_report.txt` — Report di Accuratezza

**Descrizione**: Contiene il valore di accuratezza nel validation set per ogni iterazione dell'ottimizzazione.

**Formato**: Un numero float per riga (range 0.0-1.0), ordinati cronologicamente.

**Intervallo tipico**: Nei primi step esperimenti mostrano accuratezza 0.4-0.6 (underfitting), con progressione a 0.8-0.95 negli step finali.

**Esempio**:
```
0.485
0.512
0.548
0.629
0.734
0.823
0.891
0.934
```

#### 14.1.3 `score_report.txt` — Score dell'Obiettivo

**Descrizione**: Registra il valore della funzione obiettivo (negato per compatibilità con maximizer) per ogni iterazione.

**Formato**: Un numero float negativo per riga, dove il valore assoluto rappresenta l'accuratezza negata per l'ottimizzazione.

**Utilizzo**: Utilizzato da scikit-optimize per guida bayesiana; la relazione con `acc_report.txt` è: `score = -accuracy` se non sono presenti moduli.

**Esempio**:
```
-0.485
-0.512
-0.548
-0.629
-0.734
-0.823
-0.891
-0.934
```

#### 14.1.4 `params_report.txt` — Parametri di Rete

**Descrizione**: Registra il numero di parametri (o FLOPs) della rete neurale risultante per ogni iterazione.

**Formato**: Un numero intero per riga, rappresentante il numero totale di parametri o operazioni floating-point.

**Intervallo tipico**: Da 1-10 milioni (reti piccole) a 100-200 milioni (reti grandi) in base alla configurazione sperimentale.

**Utilizzo**: Correlazione con accuratezza; generalmente reti più grandi hanno accuratezza migliore ma costo computazionale più alto.

**Esempio**:
```
1250000
2100000
3450000
5680000
8900000
12300000
16750000
24500000
```

#### 14.1.5 `diagnosis_symbolic_logs.txt` — Diagnosi Simboliche

**Descrizione**: Contiene le diagnosi simboliche generate dal modello probabilistico ProbLog, identificando i problemi di training nella rete neurale.

**Formato**: Una lista Python di stringhe per riga, dove ogni stringa è un tipo di diagnosi.

**Tipi di diagnosi**:
- `underfitting`: La rete ha capacità insufficiente per il compito (accuratezza < threshold)
- `overfitting`: Alta accuratezza di training ma bassa di validation (gap importante)
- `floating_loss`: Loss non converge o fluttua (problema di training)
- `need_skip`: Suggerimento di aggiungere skip connections per migliorare flusso del gradiente

**Evoluzione tipica**: Primi step dominati da `underfitting`, transizione a `overfitting` negli step finali, presenza occasionale di `floating_loss` in configurazioni problematiche.

**Esempio**:
```python
['underfitting']
['underfitting', 'floating_loss']
['underfitting']
['underfitting']
['overfitting']
['overfitting', 'need_skip']
['floating_loss']
['overfitting']
```

#### 14.1.6 `tuning_symbolic_logs.txt` — Azioni di Tuning Proposte

**Descrizione**: Contiene le azioni di tuning simboliche proposte dal sistema per affrontare le diagnosi identified in ogni iterazione.

**Formato**: Una lista Python di stringhe per riga, dove ogni stringa è un tipo di azione di tuning.

**Tipi di azioni**:
- `data_augmentation`: Aggiungere data augmentation per combattere overfitting
- `add_residual`: Aggiungere skip connections
- `inc_conv_layers`: Incrementare numero di layer convoluzionali
- `inc_batch_size`: Aumentare batch size
- `decr_lr`: Diminuire learning rate
- `dec_dropout`: Decrementare dropout
- `new_fc_layers`: Aggiungere layer densi aggiuntivi
- `new_conv_block`: Aggiungere blocchi convoluzionali
- `inc_neurons`: Incrementare numero di neuroni
- `reg_l2`: Applicare regolarizzazione L2
- `remove_reg_l2`: Rimuovere regolarizzazione L2

**Evoluzione**: Primi step actions singole (es. `['inc_neurons']`), step intermedi con 2 azioni (es. `['data_augmentation', 'add_residual']`), step finali con 2-3 azioni complesse.

**Esempio**:
```python
['inc_neurons']
['inc_neurons', 'inc_conv_layers']
['inc_batch_size', 'decr_lr']
['data_augmentation', 'add_residual']
['dec_dropout']
['data_augmentation', 'inc_neurons', 'add_residual']
['decr_lr']
['new_fc_layers', 'dec_dropout']
```

#### 14.1.7 `evidence.txt` — Evidenza per Apprendimento da Interpretazioni (LFI)

**Descrizione**: Registra coppie (azione, risultato) che documentano l'efficacia di ogni azione di tuning intrapresa. Utilizzato dal sistema Learning From Interpretations (LFI) per adattare i pesi delle regole simboliche.

**Formato**: Una tupla Python per riga con struttura: `[(action(metodo, diagnosi), successo), ...]`

**Campo `successo`**: 
- `True`: L'azione ha risolto la diagnosi (accuratezza migliorata)
- `False`: L'azione non ha risolto la diagnosi (accuratezza non migliorata)

**Utilizzo**: L'algoritmo di LFI analizza questo file per imparare quali combinazioni (azione, diagnosi) sono più efficaci, adattando i pesi delle regole probabilistiche per future iterazioni.

**Esempio**:
```python
[(action(inc_neurons, underfitting), False), (action(inc_conv_layers, underfitting), False)]
[(action(inc_batch_size, floating_loss), True), (action(decr_lr, floating_loss), True)]
[(action(data_augmentation, overfitting), False)]
[(action(add_residual, need_skip), True), (action(dec_dropout, overfitting), False)]
```

### 14.2 Struttura Completa della Directory

La struttura tipica di una directory di risultati è la seguente:

```
results/
├── cifar10/
│   └── 26_03_20_15_75562_cifar10_filtered_42/  # Identificativo esperimento
│       ├── algorithm_logs/  # Directory principale dei log
│       │   ├── hyper-neural.txt          # 28 iperparametri (una riga per iterazione)
│       │   ├── acc_report.txt            # 28 accuratezze (0.485 → 0.934)
│       │   ├── score_report.txt          # 28 score negativi (-0.934 → -0.485)
│       │   ├── params_report.txt         # 28 conteggi parametri network
│       │   ├── diagnosis_symbolic_logs.txt # 28 diagnosi symboliche
│       │   ├── tuning_symbolic_logs.txt   # 28 azioni di tuning proposte
│       │   └── evidence.txt               # 27 coppie (azione, risultato)
│       ├── best_models/
│       │   ├── best_accuracy_model.pth    # Miglior modello per accuratezza
│       │   └── best_flops_model.pth       # Miglior modello per FLOPs (se multi-obiettivo)
│       ├── config.yaml                    # Configurazione dell'esperimento
│       ├── results_summary.txt             # Riassunto finale accuratezza/FLOPs
│       └── training_logs.txt               # Log di training di ciascun step
```

**Note sulla Numerazione**:
- A ogni iterazione di ottimizzazione bayesiana, una nuova riga viene aggiunta ai 7 file log
- `evidence.txt` contiene n-1 righe (genera l'evidenza DOPO il test della configurazione)
- File sono **text-based** per facilità di parsing e debugging manuale

---

## 15. Debug Mode

### Definizione

Il **Debug Mode** è una modalità operativa che permette di testare la pipeline di tuning senza eseguire il training effettivo della rete neurale. Quando il nome dell'esperimento contiene la parola **"debug"** (case-insensitive), il sistema genera valori casuali di loss e accuracy per ogni epoca, simulando un training completo.

### Utilizzo

Per attivare il Debug Mode, includere la parola "debug" nel nome dell'esperimento:

```bash
# Attiva Debug Mode
python symbolic_tuner.py --name debug_test_experiment --backend tf --epoch 10 --eval 5

# Attiva Debug Mode col backend PyTorch
python symbolic_tuner.py --name debug_torch_test --backend torch --epoch 10 --eval 5

# Disattiva Debug Mode (niente "debug" nel nome)
python symbolic_tuner.py --name experiment_v1 --backend tf --epoch 10 --eval 5
```

### Implementazione per Backend

#### PyTorch (`pytorch_implementation/neural_network.py`)

Il check del debug mode avviene **dentro il loop di epoch**, permettendo alternanza tra debug e training reale:

```python
for epoch in range(self.exp_cfg.epochs):
    if 'debug' in self.exp_cfg.name.lower():
        # Genera valori casuali per questa epoca
        train_loss = float(np.random.uniform(0.5, 2.0))
        val_loss = float(np.random.uniform(0.5, 2.0))
        train_acc = float(np.random.uniform(0.4, 0.95))
        val_acc = float(np.random.uniform(0.4, 0.95))
    else:
        # Esegue il training reale
        self.model.train()
        # ... forward pass, backward pass, evaluation
```

**Vantaggi**:
- ✅ Flessibilità: Permette test rapidi senza perdere la struttura
- ✅ Overhead minimo: Solo un if per epoca
- ✅ Mantiene la compatibility con early stopping e learning rate scheduling

**Output console**:
```
Epoch 1: loss=1.2345, val_loss=1.5678, acc=0.6234, val_acc=0.5890
Epoch 2: loss=1.0987, val_loss=1.4321, acc=0.6789, val_acc=0.6234
```

#### TensorFlow (`tensorflow_implementation/neural_network.py`)

Il check del debug mode avviene **prima del training**, creando due branch separati:

```python
if 'debug' in self.exp_cfg.name.lower():
    # Genera tutte le epoche con valori casuali
    for epoch in range(self.epochs):
        train_loss = float(np.random.uniform(0.5, 2.0))
        # ... genera valori e stampa nel formato TensorFlow
else:
    # Esegue il training reale
    history = self.model.model.fit(...)
```

**Vantaggi**:
- ✅ Chiarezza strutturale: Separazione netta tra debug e training
- ✅ Realismo output: Include batch count e timing metriche
- ✅ Completezza: Gestisce tutti i parametri di configurazione

**Output console** (formato TensorFlow reale):
```
Epoch 1/100
17/17 - 0s - loss: 1.2345 - accuracy: 0.6234 - val_loss: 1.5678 - val_accuracy: 0.5890 - 175ms/epoch - 10ms/step
Epoch 2/100
17/17 - 0s - loss: 1.0987 - accuracy: 0.6789 - val_loss: 1.4321 - val_accuracy: 0.6234 - 180ms/epoch - 11ms/step
```

### Distribuzione dei Valori Casuali

Entrambe le implementazioni usano `np.random.uniform()` con i seguenti intervalli:

| Parametro | Min | Max | Motivo |
|-----------|-----|-----|--------|
| Loss (train/val) | 0.5 | 2.0 | Range realistico per classificazione su CIFAR-10/ImageNet |
| Accuracy (train/val) | 0.4 | 0.95 | Range corrispondente a early-stage training |
| Epoch time (ms) | 150 | 200 | Simulazione realistica di timing hardware |

### Setup Configurato Automaticamente

**Prima** del debug check, entrambe le implementazioni configurano comunque:

1. **Ottimizzatore** (SGD, Adam, RMSprop)
2. **Loss function** (CrossEntropyLoss per PyTorch, categorical_crossentropy per TF)
3. **Learning rate scheduler** (ReduceLROnPlateau) – anche se non utilizzato
4. **Data loaders/preparation** (DataLoader per PyTorch, X_train/X_test per TF)

Questo garantisce che il `model.optimizer` sia sempre disponibile per il controller, anche in debug mode.

### Casi d'Uso Principali

Il Debug Mode è utile per:

- 🧪 **Test rapidi della pipeline**: Verificare il flusso senza attendere il training
- ⚡ **Validazione della struttura del codice**: Controllare che logging, callbacks e risultati funzionino
- 📊 **Verifica della logica di diagnosi**: Testare il raginonamento simbolico e le azioni proposte
- 🔍 **Debug del controller**: Verificare le operazioni senza attendere epoche di training
- 📈 **Profiling dell'infrastruttura**: Analizzare overhead di logging, saving, etc. senza calcolo pesante


---

## 16. Dipendenze

```
tensorflow==2.15          # Backend TF (opzionale se si usa PyTorch)
scikit-optimize>=0.10.2   # Bayesian Optimization
matplotlib>=3.9.0         # Plotting
pytest>=8.3.4             # Testing
problog>=2.2.6            # Ragionamento simbolico probabilistico
tqdm==4.67.1              # Barre di progresso
PyYAML==6.0.3             # Configurazione
pandas==2.3.3             # Analisi risultati
```

Per PyTorch, è necessario installare separatamente:
```
torch>=2.0
torchvision
```

---

## 17. Riferimenti Bibliografici

1. Fraccaroli, M., Lamma, E., & Riguzzi, F. (2022). *Symbolic DNN-tuner*. Machine Learning, 111(2), 625–650. [DOI: 10.1007/s10994-021-06097-1](https://link.springer.com/article/10.1007/s10994-021-06097-1)

2. Fraccaroli, M., Lamma, E., & Riguzzi, F. (2022). *Symbolic DNN-Tuner: A Python and ProbLog-based system for optimizing Deep Neural Networks hyperparameters*. SoftwareX, 17, 100957. [DOI: 10.1016/j.softx.2021.100957](https://www.sciencedirect.com/science/article/pii/S2352711021001825)
