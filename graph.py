"""
Analyze the log files contained in algorithm_logs and plot each of them:
        - accuracy, score, and params as line plots or similar
        - evidence: count how many times each action succeeded and failed (bar plot)
        - diagnosis and tuning: it may be useful to show how many times each problem and
        solution were found/applied (bar plot), and also find a way to show, for each
        iteration, which problems were found and which solutions were applied.

Usage:
    python 1-grafici.py

The user will be asked to select the parent folder containing the experiments.
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import pandas as pd

# Try to import tkinter, fallback if not available
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = False
except ImportError:
    TKINTER_AVAILABLE = False

def select_parent_directory() -> Optional[Path]:
    """
    Ask the user to select the parent folder containing the experiments.
    
    Returns:
        Path of the selected folder or None if cancelled
    """
    if TKINTER_AVAILABLE:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        directory = filedialog.askdirectory(
            title="Select the parent folder containing the experiments",
            initialdir=os.path.expand_user("~")
        )
        
        return Path(directory) if directory else None
    else:
        # Fallback: ask via command line
        print("GUI not available, please enter the path manually")
        path_input = input("Path of the parent folder containing the experiments: ").strip()
        if path_input and Path(path_input).exists():
            return Path(path_input)
        return None


@dataclass
class ExperimentResult:
    """Represents a single iteration of an experiment"""
    iteration: int
    accuracy: Optional[float]
    flops: Optional[float]
    nparams: Optional[float]
    latency: Optional[float]
    hw_cost: Optional[float]
    hw_total_cost: Optional[float]
    hw_config: Optional[str]
    evidence: Optional[Tuple[Tuple[str, str], bool]]        # New field to store evidence data
    diagnosis: Optional[List[str]]                          # New field to store diagnoses found at each iteration
    tuning: Optional[List[str]]                             # New field to store tuning solutions applied at each iteration
    score: Optional[float] = None  # Calculated as -accuracy if no modules are present
    # hyperparams: Optional[Dict[str, Any]]

class ResultsAnalyzer:
    """Analyzer for experiment results"""
    
    def __init__(self, experiment_dir: str):
        """
        Initialize the analyzer.
        
        Args:
            experiment_dir: Experiment folder containing algorithm_logs/
        """
        self.experiment_dir = Path(experiment_dir)
        self.algorithm_logs_dir = self.experiment_dir / "algorithm_logs"
        self.results: List[ExperimentResult] = []
        self.has_flops_module = False
        self.has_hardware_module = False
        self.config: Dict[str, Any] = {}  # Experiment configuration

    def load_results(self) -> bool:
        """
        Load all results from log files.
        
        Returns:
            True if logs were loaded successfully, False otherwise
        """
        # Load config.yaml file
        self._load_config_yaml()
        
        if not self.algorithm_logs_dir.exists():
            print(f"  algorithm_logs folder not found in {self.experiment_dir}")
            return False
        
        # Load accuracies
        accuracies = self._load_accuracies()
        if not accuracies:
            print(f"  No acc_report.txt file found")
            return False
        
        scores = self._load_scores()
        if scores:
            print(f"  score_report.txt file found, will use scores calculated during training")
        else:
            print(f"  No score_report.txt file found, will use -accuracy as approximation")
        
        # Load hyperparameters
        # hyperparams_list = self._load_hyperparams()
        
        # Load FLOPS data
        flops_data = self._load_flops_data()
        if flops_data:
            self.has_flops_module = True
        
        # Load hardware data
        hw_data = self._load_hardware_data()
        if hw_data:
            self.has_hardware_module = True

        evidence_data = self._load_evidence_data()      # Load evidence data

        diagnosis_data = self._load_diagnosis_data()      # Load diagnosis data (if available)

        tuning_data = self._load_tuning_data()      # Load tuning data (if available)
        
        # Combine the data
        max_iterations = max(
            len(accuracies),
            # len(hyperparams_list),
            len(flops_data) if flops_data else 0,
            len(hw_data) if hw_data else 0,
            len(evidence_data) if evidence_data else 0,      # Also consider evidence data length when determining how many iterations to analyze
            len(diagnosis_data) if diagnosis_data else 0,     # Also consider diagnosis data length when determining how many iterations to analyze
            len(tuning_data) if tuning_data else 0           # Also consider tuning data length when determining how many iterations to analyze
        )
        
        for i in range(max_iterations):
            result = ExperimentResult(
                iteration=i + 1,
                accuracy=accuracies[i] if i < len(accuracies) else None,
                flops=flops_data[i][1] if flops_data and i < len(flops_data) else None,
                nparams=flops_data[i][0] if flops_data and i < len(flops_data) else None,
                latency=hw_data[i][0] if hw_data and i < len(hw_data) else None,
                hw_cost=hw_data[i][1] if hw_data and i < len(hw_data) else None,
                hw_total_cost=hw_data[i][2] if hw_data and i < len(hw_data) else None,
                hw_config=hw_data[i][3] if hw_data and i < len(hw_data) else None,
                score=scores[i] if scores and i < len(scores) else None,
                # hyperparams=hyperparams_list[i] if i < len(hyperparams_list) else None,
                evidence=evidence_data[i] if evidence_data and i < len(evidence_data) else None,     # Add evidence data to the result
                diagnosis=diagnosis_data[i] if diagnosis_data and i < len(diagnosis_data) else None,     # Add diagnosis data to the result
                tuning=tuning_data[i] if tuning_data and i < len(tuning_data) else None               # Add tuning data to the result
            )
            
            # Calculate score: -accuracy if no modules are present
            if result.score is None:
                if result.accuracy is not None:
                    result.score = -result.accuracy
                else:
                    result.score = None
            
            self.results.append(result)
        
        return True
    
    def _load_config_yaml(self) -> None:
        """Load configuration from config.yaml file"""
        config_file = self.experiment_dir / "config.yaml"
        if not config_file.exists():
            print(f"  config.yaml file not found in {self.experiment_dir}")
            self.config = {}
            return
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"  Error reading config.yaml: {e}")
            self.config = {}

    def _load_accuracies(self) -> List[Optional[float]]:
        """Load accuracies from acc_report.txt file"""
        acc_file = self.algorithm_logs_dir / "acc_report.txt"
        if not acc_file.exists():
            return []
        
        accuracies = []
        try:
            with open(acc_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line == "None" or line == "":
                        accuracies.append(None)
                    else:
                        accuracies.append(float(line))
        except Exception as e:
            print(f"  Error reading {acc_file}: {e}")
            return []
        
        return accuracies
    
    def _load_scores(self) -> List[Optional[float]]:
        """Load scores from score_report.txt file"""
        score_file = self.algorithm_logs_dir / "score_report.txt"
        if not score_file.exists():
            return []
        
        scores = []
        try:
            with open(score_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line == "None" or line == "":
                        scores.append(None)
                    else:
                        scores.append(float(line))
        except Exception as e:
            print(f"  Error reading {score_file}: {e}")
            return []
        
        return scores
    
    def _load_flops_data(self) -> Optional[List[Tuple[float, float]]]:
        """Load FLOPS data from flops_report.txt file"""
        flops_file = self.algorithm_logs_dir / "flops_report.txt"
        if not flops_file.exists():
            flops_file = self.algorithm_logs_dir / "params_report.txt"
            if not flops_file.exists():
                flops_file = self.experiment_dir / "flops_report.txt"
                if not flops_file.exists():
                    return None

        flops_data = []
        try:
            with open(flops_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(' ')
                    if len(parts) >= 2:
                        flops_data.append((float(parts[0]), float(parts[1])))
                    else:
                        # If it's a file with only params, add None for FLOPS
                        flops_data.append((float(parts[0]), None))
        except Exception as e:
            print(f"  Error reading {flops_file}: {e}")
            return None
        
        return flops_data if flops_data else None
    
    def _load_hardware_data(self) -> Optional[List[Tuple[float, float, float, str]]]:
        """Load hardware data from hardware_report.txt file"""
        hw_file = self.algorithm_logs_dir / "hardware_report.txt"
        if not hw_file.exists():
            return None
        
        hw_data = []
        try:
            with open(hw_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    if len(parts) >= 4:
                        hw_data.append((
                            float(parts[0]),  # latency
                            float(parts[1]),  # cost
                            float(parts[2]),  # total_cost
                            parts[3].strip()   # config
                        ))
        except Exception as e:
            print(f"  Error reading {hw_file}: {e}")
            return None
        
        return hw_data if hw_data else None
    
    # !!! NEW FUNCTION TO LOAD EVIDENCE DATA
    def _load_evidence_data(self) -> Optional[List[Tuple[Tuple[str, str], bool]]]:
        """Load evidence data from evidence.txt file"""
        evidence_file = self.algorithm_logs_dir / "evidence.txt"
        if not evidence_file.exists():
            return None
        
        evidence_data = []
        try:
            with open(evidence_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    line = line.replace("[","").replace("]","").replace("action", "").replace("(", "").replace(")", "").replace(" ", "")    # Remove extra words and symbols to keep only the data
                    parts = line.split(',')
                    if len(parts) >= 0:
                        for i in range (0, len(parts)-2, 3):    # The data comes in triples, so iterate by 3 to read it correctly
                            action = (parts[i+0], parts[i+1])
                            success = parts[i+2].lower() == 'true'  # Convert the string to a boolean
                            evidence_data.append((action, success))

        except Exception as e:
            print(f"  Error reading {evidence_file}: {e}")
            return None
        
        return evidence_data if evidence_data else None
    
    # !!! NEW FUNCTION TO LOAD DIAGNOSIS DATA
    def _load_diagnosis_data(self) -> Optional[List[List[str]]]:
        """Load diagnosis data from diagnosis_symbolic_logs.txt file"""
        diagnosis_file = self.algorithm_logs_dir / "diagnosis_symbolic_logs.txt"
        if not diagnosis_file.exists():
            return None
        
        diagnosis_data = []
        try:
            with open(diagnosis_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    line = line.replace("[","").replace("]","").replace("'", "").replace(" ", "")    # Remove extra symbols to keep only the data
                    diagnoses = line.split(',')  # Each diagnosis is separated by a comma, so split the line on that character
                    diagnosis_data.append(diagnoses)

        except Exception as e:
            print(f"  Error reading {diagnosis_file}: {e}")
            return None
        
        return diagnosis_data if diagnosis_data else None
    
    # !!! NEW FUNCTION TO LOAD TUNING DATA
    def _load_tuning_data(self) -> Optional[List[List[str]]]:
        """Load tuning data from tuning_symbolic_logs.txt file"""
        tuning_file = self.algorithm_logs_dir / "tuning_symbolic_logs.txt"
        if not tuning_file.exists():
            return None
        
        tuning_data = []
        try:
            with open(tuning_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    line = line.replace("[","").replace("]","").replace("'", "").replace(" ", "")    # Remove extra symbols to keep only the data
                    tunings = line.split(',')  # Each tuning solution is separated by a comma, so split the line on that character
                    tuning_data.append(tunings)

        except Exception as e:
            print(f"  Error reading {tuning_file}: {e}")
            return None
        
        return tuning_data if tuning_data else None


def analyze_all_experiments(parent_dir: Path, output_dir: Optional[Path] = None):
    """
    Analyze all experiments in the parent folder.
    
    Args:
        parent_dir: Parent folder containing the experiments
        output_dir: Output folder (default: parent_dir)
    """
    if output_dir is None:
        output_dir = parent_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all subfolders with algorithm_logs
    experiments = []
    for item in parent_dir.iterdir():
        if item.is_dir() and (item / "algorithm_logs").exists():
            experiments.append(item)
    
    if not experiments:
        print(f"No experiments found in {parent_dir}")
        return
    
    print(f"\nFound {len(experiments)} experiments\n")
    
    # Analyze each experiment
    summary_data = []
    
    for exp_dir in sorted(experiments):
        print(f"📈 Analyzing: {exp_dir.name}")
        
        analyzer = ResultsAnalyzer(exp_dir)
        if not analyzer.load_results():
            continue

    # Begin modifications   !!! 
        # Create three charts for each experiment in the same figure: accuracy, score, and nparams (if available)
        # First check which data is available to decide how many charts to create 
        num_plots = 0
        plotAccuracy = False
        plotScore = False
        plotParams = False

        for i in range(len(analyzer.results)):  # Check whether there is at least one valid accuracy point before creating the accuracy chart
            if analyzer.results[i].accuracy is not None:
                num_plots += 1
                plotAccuracy = True
                break
        for i in range(len(analyzer.results)):
            if analyzer.results[i].score is not None and analyzer.results[i].score < 0:
                num_plots += 1
                plotScore = True
                break
        for i in range(len(analyzer.results)):
            if analyzer.results[i].nparams is not None:
                num_plots += 1
                plotParams = True
                break

        print(f"  - {num_plots} line plots to generate for {exp_dir.name}")
        
        if num_plots == 0:
            continue

        fig, axes = plt.subplots(1,num_plots, figsize=(10 * num_plots, 6))      # Adapt the figure size to the number of charts to create
        # Line chart for accuracy
        idx = 0
        if plotAccuracy:
            axes[idx].plot(
                [r.iteration for r in analyzer.results if r.accuracy is not None],  #asse x sono le iterazioni
                [r.accuracy for r in analyzer.results if r.accuracy is not None],   #asse y sono le accuratezze, filtro solo quelle non None
                label=exp_dir.name, color='red'
            )
            axes[idx].set_title(f"Accuracy - {exp_dir.name}")
            axes[idx].set_xlabel("Iteration")
            axes[idx].set_ylabel("Accuracy")
            axes[idx].grid()
            axes[idx].plot()
            idx += 1

        # Line chart for score
        if plotScore:
            axes[idx].plot(
                [r.iteration for r in analyzer.results if r.score is not None and r.score<0],   #asse x sono le iterazioni
                [r.score for r in analyzer.results if r.score is not None and r.score<0],       #asse y sono gli score, filtro solo quelli non None e negativi 
                label=exp_dir.name, color='blue'
            )
            axes[idx].set_title(f"Score - {exp_dir.name}")
            axes[idx].set_xlabel("Iteration")
            axes[idx].set_ylabel("Score")
            axes[idx].grid()
            axes[idx].plot()
            idx += 1

        # Line chart for the number of parameters
        if plotParams:
            axes[idx].plot(
                [r.iteration for r in analyzer.results if r.nparams is not None],       #asse x sono le iterazioni
                [r.nparams for r in analyzer.results if r.nparams is not None],         #asse y sono il numero di parametri, filtro solo quelli non None
                label=exp_dir.name, color='green'
            )
            axes[idx].set_title(f"Number of Parameters - {exp_dir.name}")
            axes[idx].set_xlabel("Iteration")
            axes[idx].set_ylabel("Number of Parameters")
            axes[idx].grid()
            axes[idx].plot()

        plt.tight_layout()
        plt.savefig(output_dir / f"{exp_dir.name}_graphs.png")
        plt.close()

        # Bar chart for evidence data
        dati_evidence = []          # List used to store evidence data to plot  
        for r in analyzer.results:
            if r.evidence is not None:
                dati_evidence.append(r.evidence)

        if dati_evidence:       # If there is evidence data to plot, generate the bar chart
            print (f"  - Generating evidence bar plot for {exp_dir.name} ")
            fig, ax = plt.subplots(figsize=(15, 8))
            df = pd.DataFrame(dati_evidence, columns=['Tupla', 'Condizione'])       # Create a DataFrame with two columns: one for the tuple (action) and one for the condition (success/failure) to help with axes and calculations

            df['Etichetta'] = df['Tupla'].apply(lambda x: "-".join(map(str, x)))    # Create a new "Etichetta" column by joining the tuple values into a string so it can be used as the bar chart x-axis label

            conteggi = pd.crosstab(df['Etichetta'], df['Condizione'])           # Use crosstab to count how many times each label succeeded (True) and failed (False), producing a table with labels as rows and conditions as columns

            conteggi.plot(kind='bar', color=['red', 'green'], ax=ax) # Generate a bar plot: red for False, green for True

            ax.set_title('Conteggio True vs False per Categoria')
            ax.set_xlabel('Categorie (Tuple)')
            ax.set_ylabel('Frequenza')
            ax.set_xticklabels(conteggi.index, rotation=45, ha='right', rotation_mode='anchor') # Rotate and align the x-axis labels
            ax.legend(['Falso', 'Vero'])
            fig.tight_layout()
            fig.savefig(output_dir / f"{exp_dir.name}_evidence.png")
            plt.close(fig)

        # Plot the frequency of the methods used (regardless of success) to see which one is used most
        print (f"  - Generating method frequency bar plot for {exp_dir.name} ")
        grafici_tuning = 0
        grafico_evidence = False
        grafico_t=False

        if any(r.tuning is not None for r in analyzer.results):
            grafici_tuning+=1
            grafico_t=True
        if any(r.evidence is not None for r in analyzer.results):
            grafici_tuning+=1
            grafico_evidence = True

        if grafici_tuning >0:
            fig, ax = plt.subplots(1,grafici_tuning, figsize=(10 * grafici_tuning, 6))
            if grafici_tuning == 1:
                ax = [ax]
            idx=0
            if grafico_evidence:
                df['Metodo'] = df['Tupla'].apply(lambda x: x[0])   # Create a new "Metodo" column by extracting only the first part of the tuple, which represents the action taken, so it can be used as the bar chart category
                frequenza_azioni = df['Metodo'].value_counts()     # Calculate the frequency of each method using value_counts, producing a series with methods as the index and frequencies as values
                frequenza_azioni.plot(kind='bar', color='blue', ax=ax[idx])  # Generate a bar plot with the method frequencies, using blue
                ax[idx].set_title('Frequenza dei Metodi Intrapresi')
                ax[idx].set_xlabel('Metodo Intrapreso')
                ax[idx].set_ylabel('Frequenza')
                ax[idx].set_xticklabels(frequenza_azioni.index, rotation=45, ha='right', rotation_mode='anchor')
                idx+=1
            if grafico_t:
                tuning_data = []      # List used to store diagnoses found in diagnosis_symbolic_logs
                for r in analyzer.results:
                    if r.tuning is not None:
                        tuning_data.extend(r.tuning)   # Add all tuning solutions found in this iteration to the main list

                df_tuning = pd.DataFrame(tuning_data, columns=['Tuning'])   # Create a DataFrame with a "Tuning" column for tuning_symbolic_logs data
                frequenza_tuning = df_tuning['Tuning'].value_counts()        # Calculate the frequency of each diagnosis using value_counts, producing a series with diagnoses as the index and frequencies as values
                frequenza_tuning.plot(kind='bar', color='orange', ax=ax[idx])       # Generate a bar plot with the diagnosis frequencies, using orange
                ax[idx].set_title('Frequenza dei Tuning (tuning_symbolic_logs)')
                ax[idx].set_xlabel('Tuning Applicato')
                ax[idx].set_ylabel('Frequenza')
                ax[idx].set_xticklabels(frequenza_tuning.index, rotation=45, ha='right', rotation_mode='anchor')

            fig.tight_layout()
            fig.savefig(output_dir / f"{exp_dir.name}_action_frequency.png")
            plt.close(fig)

        # Plot the specific diagnoses from both evidence data and diagnosis_symbolic_logs
        print (f"  - Generating diagnosis frequency bar plot for {exp_dir.name} ")
        grafici_diagnosi = 0
        grafico_evidence = False
        grafico_d=0

        if any(r.diagnosis is not None for r in analyzer.results):
            grafici_diagnosi+=1
            grafico_d=True
        if any(r.evidence is not None for r in analyzer.results):
            grafici_diagnosi+=1
            grafico_evidence = True
        if grafici_diagnosi >0:
            
            fig, ax = plt.subplots(1,grafici_diagnosi, figsize=(10 * grafici_diagnosi, 6))
            if grafici_diagnosi == 1:
                ax = [ax]
            idx=0
            if grafico_evidence:
                df['Diagnosi'] = df['Tupla'].apply(lambda x: x[1])   # Create a new "Diagnosi" column by extracting only the second part of the tuple, which represents the diagnosis, so it can be used as the bar chart category
                frequenza_azioni = df['Diagnosi'].value_counts()     # Calculate the frequency of each diagnosis using value_counts, producing a series with diagnoses as the index and frequencies as values
                frequenza_azioni.plot(kind='bar', color='blue', ax=ax[0])  # Generate a bar plot with the diagnosis frequencies, using blue
                ax[0].set_title('Frequenza delle Diagnosi')
                ax[0].set_xlabel('Diagnosi')
                ax[0].set_ylabel('Frequenza')
                ax[0].set_xticklabels(frequenza_azioni.index, rotation=45, ha='right', rotation_mode='anchor')
                idx=1

            # If diagnosis_symbolic_logs data is also available, generate a second bar chart to show the frequency of diagnoses found in those logs
            if grafico_d:   # Check whether diagnosis_symbolic_logs data is available in at least one iteration
                diagnosis_data = []      # List used to store diagnoses found in diagnosis_symbolic_logs
                for r in analyzer.results:
                    if r.diagnosis is not None:
                        diagnosis_data.extend(r.diagnosis)   # Add all diagnoses found in this iteration to the main list

                df_diagnosis = pd.DataFrame(diagnosis_data, columns=['Diagnosi'])   # Create a DataFrame with a "Diagnosi" column for diagnosis_symbolic_logs data
                frequenza_diagnosi = df_diagnosis['Diagnosi'].value_counts()        # Calculate the frequency of each diagnosis using value_counts, producing a series with diagnoses as the index and frequencies as values
                frequenza_diagnosi.plot(kind='bar', color='orange', ax=ax[idx])       # Generate a bar plot with the diagnosis frequencies, using orange
                ax[idx].set_title('Frequenza delle Diagnosi (diagnosis_symbolic_logs)')
                ax[idx].set_xlabel('Diagnosi')
                ax[idx].set_ylabel('Frequenza')
                ax[idx].set_xticklabels(frequenza_diagnosi.index, rotation=45, ha='right', rotation_mode='anchor')

            fig.tight_layout()
            fig.savefig(output_dir / f"{exp_dir.name}_diagnosis_frequency.png")
            plt.close(fig)

            # For each iteration, show which diagnoses were found and which tuning solutions were applied, creating a plot with iterations on the x-axis and diagnoses/solutions on the y-axis
            # Plot the diagnoses first and then the tuning solutions, using different colors to distinguish them visually
            print (f"  - Generating diagnosis and tuning timeline plot for {exp_dir.name} ")
            fig, ax = plt.subplots(figsize=(15, 8))
            timeline_data_diagnosis = []   # List used to store data for the diagnosis timeline
            timeline_data_tuning = []   # List used to store data for the tuning timeline
            for r in analyzer.results:
                if r.diagnosis is not None:
                    for diag in r.diagnosis:
                        timeline_data_diagnosis.append((r.iteration, diag))   # Add a tuple with iteration and diagnosis 
                if r.tuning is not None:
                    for tune in r.tuning:
                        timeline_data_tuning.append((r.iteration, tune))     # Add a tuple with iteration and tuning solution 
            if timeline_data_diagnosis or timeline_data_tuning:   # Check whether there is timeline data to plot
                df_diagnosis = pd.DataFrame(timeline_data_diagnosis, columns=['Iteration', 'Evento'])   # Create a DataFrame with columns for iteration and diagnosis event
                df_tuning = pd.DataFrame(timeline_data_tuning, columns=['Iteration', 'Evento'])   # Create a DataFrame with columns for iteration and tuning event
                df=df_diagnosis.copy()
                df = pd.concat([df, df_tuning], ignore_index=True)   # Merge the two DataFrames so diagnosis and tuning can be plotted in the same chart
                df['Color'] = df['Evento'].apply(lambda x: 'blue' if x in [diag for r in analyzer.results for diag in (r.diagnosis or [])] else 'orange')  # Create a "Color" column to distinguish diagnosis and tuning visually
                ax.scatter([], [], color='blue', label='Diagnosis')   # Add a dummy point for the diagnosis legend
                ax.scatter([], [], color='orange', label='Tuning')    # Add a dummy point for the tuning legend
                ax.legend()
                for _, row in df.iterrows():
                    ax.scatter(row['Iteration'], row['Evento'], color=row['Color'])   # Add a point to the chart for each event, using the matching color
                ax.set_title('Timeline of Diagnoses and Tuning')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Event')
                ax.grid()
                fig.tight_layout()
                fig.savefig(output_dir / f"{exp_dir.name}_diagnosis_tuning_timeline.png")
                plt.close(fig)

def main():
    """Main function"""
    print("\n" + "="*70)
    print("  EXPERIMENT RESULTS ANALYZER - Symbolic DNN Tuner")
    print("="*70 + "\n")
    
    # Check if arguments have been passed
    if len(sys.argv) > 1:
        # Batch mode: use command line arguments
        parent_dir = Path(sys.argv[1])
        if not parent_dir.exists():
            print(f"Folder not found: {parent_dir}")
            return
        
        output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Input folder: {parent_dir}")
        if output_dir:
            print(f"Output folder: {output_dir}\n")
    else:
        # Interactive mode
        parent_dir = select_parent_directory()
        if parent_dir is None:
            print("No folder selected")
            return
        
        print(f"\nSelected folder: {parent_dir}\n")
        
        # Ask if save output in a different folder (only if not in stdin)
        try:
            use_custom_output = input("Save results in a different folder? (y/n): ").lower() == 'y'
            
            output_dir = None
            if use_custom_output:
                if TKINTER_AVAILABLE:
                    root = tk.Tk()
                    root.withdraw()
                    output_path = filedialog.askdirectory(
                        title="Select the output folder",
                        initialdir=str(parent_dir)
                    )
                    if output_path:
                        output_dir = Path(output_path)
                    else:
                        print("⚠ No output folder selected, using parent folder")
                else:
                    output_path = input("Output folder path: ").strip()
                    if output_path and Path(output_path).exists():
                        output_dir = Path(output_path)
                    else:
                        print("⚠ Invalid path, using parent folder")
        except EOFError:
            # If stdin is closed, use non-interactive mode
            output_dir = None
    
    # Analyze experiments
    analyze_all_experiments(parent_dir, output_dir)
    
    print("\n" + "="*70)
    print("  Analysis completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
