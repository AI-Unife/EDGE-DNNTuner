import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
            experiment_dir: Experiment folder containing algorithm_logs
        """
        self.experiment_dir = Path(experiment_dir)
        self.algorithm_logs_dir = os.path.join(self.experiment_dir, "algorithm_logs")
        self.results: List[ExperimentResult] = []
        self.has_flops_module = False
        self.has_hardware_module = False
        self.config: Dict[str, Any] = {}  # Experiment configuration
        self.exp_name = self.experiment_dir.name  # Experiment name based on the folder name

    def load_results(self) -> bool:
        """
        Load all results from log files.
        
        Returns:
            True if logs were loaded successfully, False otherwise
        """
        # Load config.yaml file
        self._load_config_yaml()
        
        if not os.path.exists(self.algorithm_logs_dir):
            return False
        
        # Load accuracies
        accuracies = self._load_accuracies()
        if not accuracies:
            return False
        
        scores = self._load_scores()
        
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
        config_file = os.path.join(self.experiment_dir, "config.yaml")
        if not os.path.exists(config_file):
            self.config = {}
            return
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        except Exception as e:
            self.config = {}

    def _load_accuracies(self) -> List[Optional[float]]:
        """Load accuracies from acc_report.txt file"""
        acc_file = os.path.join(self.algorithm_logs_dir, "acc_report.txt")
        if not os.path.exists(acc_file):
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
            return []
        
        return accuracies
    
    def _load_scores(self) -> List[Optional[float]]:
        """Load scores from score_report.txt file"""
        score_file = os.path.join(self.algorithm_logs_dir, "score_report.txt")
        if not os.path.exists(score_file):
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
            return []
        
        return scores
    
    def _load_flops_data(self) -> Optional[List[Tuple[float, float]]]:
        """Load FLOPS data from flops_report.txt file"""
        flops_file = os.path.join(self.algorithm_logs_dir, "flops_report.txt")
        if not os.path.exists(flops_file):
            flops_file = os.path.join(self.algorithm_logs_dir, "params_report.txt")
            if not os.path.exists(flops_file):
                flops_file = os.path.join(self.experiment_dir, "flops_report.txt")
                if not os.path.exists(flops_file):
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
            return None
        
        return flops_data if flops_data else None
    
    def _load_hardware_data(self) -> Optional[List[Tuple[float, float, float, str]]]:
        """Load hardware data from hardware_report.txt file"""
        hw_file = os.path.join(self.algorithm_logs_dir, "hardware_report.txt")
        if not os.path.exists(hw_file):
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
            return None
        
        return hw_data if hw_data else None
    
    # !!! NEW FUNCTION TO LOAD EVIDENCE DATA
    def _load_evidence_data(self) -> Optional[List[Tuple[Tuple[str, str], bool]]]:
        """Load evidence data from evidence.txt file"""
        evidence_file = os.path.join(self.algorithm_logs_dir, "evidence.txt")
        if not os.path.exists(evidence_file):
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
            return None
        
        return evidence_data if evidence_data else None
    
    # !!! NEW FUNCTION TO LOAD DIAGNOSIS DATA
    def _load_diagnosis_data(self) -> Optional[List[List[str]]]:
        """Load diagnosis data from diagnosis_symbolic_logs.txt file"""
        diagnosis_file = os.path.join(self.algorithm_logs_dir, "diagnosis_symbolic_logs.txt")
        if not os.path.exists(diagnosis_file):
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
            return None
        
        return diagnosis_data if diagnosis_data else None
    
    # !!! NEW FUNCTION TO LOAD TUNING DATA
    def _load_tuning_data(self) -> Optional[List[List[str]]]:
        """Load tuning data from tuning_symbolic_logs.txt file"""
        tuning_file = os.path.join(self.algorithm_logs_dir, "tuning_symbolic_logs.txt")
        if not os.path.exists(tuning_file):
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
            return None
        
        return tuning_data if tuning_data else None
    
def plotaccuracy_confronto(analyzers: list[ResultsAnalyzer], output_dir: Optional[Path] = None):
    # Create three comparison plots in the same figure: accuracy, score, and nparams (if available)
    # First check which data is available to decide how many charts to create 
    num_plots = 0
    plotAccuracy = False
    plotScore = False
    plotParams = False

    for analyzer in analyzers:
        for i in range(len(analyzer.results)):  # Check whether there is at least one valid accuracy point before creating the accuracy chart
            if analyzer.results[i].accuracy is not None:
                num_plots += 1
                plotAccuracy = True
                break
        if plotAccuracy:
            break

    for analyzer in analyzers:
        for i in range(len(analyzer.results)):
            if analyzer.results[i].score is not None and analyzer.results[i].score < 0:
                num_plots += 1
                plotScore = True
                break
        if plotScore:
            break

    for analyzer in analyzers:
        for i in range(len(analyzer.results)):
            if analyzer.results[i].nparams is not None:
                num_plots += 1
                plotParams = True
                break
        if plotParams:
            break
    
    if num_plots == 0:
        return None  # If there is nothing to plot, exit without creating a chart

    # Determine the chart titles based on the available data
    cols = sum([plotAccuracy, plotScore, plotParams])
    titles = []
    if plotAccuracy: titles.append("Accuracy")
    if plotScore: titles.append("Score")
    if plotParams: titles.append("Number of Parameters")

    # Create the subplot figure for comparison charts using Plotly
    fig = make_subplots(rows=1, cols=cols, subplot_titles=titles)
    colors = px.colors.qualitative.Plotly  # Palette standard (10 colori distinti)
    
    for i, analyzer in enumerate(analyzers):
        exp_name = analyzer.experiment_dir.name
        exp_color = colors[i % len(colors)] 
        idx = 1 
        # --- ACCURACY LOGIC ---
        if plotAccuracy:
            x_acc = [r.iteration for r in analyzer.results if r.accuracy is not None]
            y_acc = [r.accuracy for r in analyzer.results if r.accuracy is not None]
            fig.add_trace(go.Scatter(x=x_acc, y=y_acc, name=exp_name, mode='lines',line=dict(color=exp_color),
                                     legendgroup=exp_name, showlegend=True), row=1, col=idx)
            idx = idx +1  # Increment the index only if the accuracy chart was added

        # --- SCORE LOGIC  ---
        if plotScore:
            x_score = [r.iteration for r in analyzer.results if r.score is not None and r.score<0]
            y_score = [r.score for r in analyzer.results if r.score is not None and r.score<0]
            fig.add_trace(go.Scatter(x=x_score, y=y_score, name=exp_name, mode='lines',line=dict(color=exp_color),
                                     legendgroup=exp_name, showlegend=False), row=1, col=idx)
            idx += 1  # Increment the index only if the accuracy chart was added

        # --- NUMBER OF PARAMETERS LOGIC  ---
        if plotParams:
            x_par = [r.iteration for r in analyzer.results if r.nparams is not None]
            y_par = [r.nparams for r in analyzer.results if r.nparams is not None]
            fig.add_trace(go.Scatter(x=x_par, y=y_par, name=exp_name, mode='lines',line=dict(color=exp_color),
                                     legendgroup=exp_name, showlegend=False), row=1, col=idx)

    # Personalizzazione interattiva
    fig.update_layout(
        height=400, 
        width= 600 * cols,
        hovermode="x unified", # Show all data near the mouse on the same X axis
        template="plotly_white",
        autosize=False
    )
    if output_dir:
        try: 
            output_path = Path(output_dir) 
            output_path=os.path.join(output_path, f"Comparison_Accuracy_Score_Params.png")
            fig.write_image(output_path)
        #if there isn't kaleido installed, the image cannot be saved, but the function should return the figure anyway without crashing
        except Exception as e:
            print(f"Warning: Could not save the image due to error: {e}")
            return None
    return fig

def plotaccuracy(analyzer: ResultsAnalyzer, exp: Path, output_dir: Optional[Path] = None):
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
    
    if num_plots == 0:
        return None  # If there is nothing to plot, exit without creating a chart
    
    cols = sum([plotAccuracy, plotScore, plotParams])   # Count how many charts to create based on the available data
    titles = []                                         # Determine the chart titles based on the available data
    if plotAccuracy: titles.append("Accuracy")
    if plotScore: titles.append("Score")
    if plotParams: titles.append("Number of Parameters")

    # Create the subplot figure for comparison charts using Plotly
    fig = make_subplots(rows=1, cols=cols, subplot_titles=titles)
    
    idx = 1 
    # --- ACCURACY LOGIC ---
    if plotAccuracy:
        data_acc = [(r.iteration, r.accuracy, i) for i, r in enumerate(analyzer.results) if r.accuracy is not None]
        x_acc, y_acc, custom_acc = zip(*data_acc)
        fig.add_trace(go.Scatter(x=x_acc, y=y_acc,customdata=custom_acc,name="accuracy", mode='lines+markers',line=dict(color='red'),showlegend=False), row=1, col=idx)
        idx = idx +1  # Increment the index only if the accuracy chart was added

    # --- SCORE LOGIC  ---
    if plotScore:
        x_score = [r.iteration for r in analyzer.results if r.score is not None and r.score<0]
        y_score = [r.score for r in analyzer.results if r.score is not None and r.score<0]
        fig.add_trace(go.Scatter(x=x_score, y=y_score,name="score", mode='lines+markers',line=dict(color='blue'),showlegend=False), row=1, col=idx)
        idx += 1  # Increment the index only if the accuracy chart was added

    # --- NUMBER OF PARAMETERS LOGIC  ---
    if plotParams:
        x_par = [r.iteration for r in analyzer.results if r.nparams is not None]
        y_par = [r.nparams for r in analyzer.results if r.nparams is not None]
        fig.add_trace(go.Scatter(x=x_par, y=y_par,name="nparam", mode='lines+markers',line=dict(color='green'),showlegend=False), row=1, col=idx)

    # Personalizzazione interattiva
    fig.update_layout(
        height=400, 
        width=600 * cols,
        hovermode="closest", 
        clickmode="event+select",  # Enable click mode to select data points
        template="plotly_white",
        autosize=False
    )
    if output_dir:
        try:
            output_path = Path(output_dir) 
            output_path=os.path.join(output_path, f"{exp}_Accuracy_Score_Params.png")
            fig.write_image(output_path)
        except Exception as e:
            print(f"Warning: Could not save the image due to error: {e}")
            return None
    return fig

def plotevidence_confronto(analyzers, output_dir: Optional[Path] = None):
    dati_evidence = []    
    for analyzer in analyzers:
        exp_name = analyzer.experiment_dir.name 
        for r in analyzer.results:
            if r.evidence is not None:
                dati_evidence.append({
                    'Azione': "-".join(map(str, r.evidence[0])),
                    'Esito': r.evidence[1],
                    'Esperimento': exp_name
                })

    if dati_evidence:
        df = pd.DataFrame(dati_evidence)

        # Split the data by action, outcome, and experiment, then count occurrences to create a pivot table with action-outcome rows, experiment columns, and counts as values
        pivot_df = df.groupby(['Azione', 'Esito', 'Esperimento']).size().unstack(fill_value=0)
        
        # Make sure each action has both success and failure columns
        tutte_azioni = df['Azione'].unique()
        nuovo_indice = pd.MultiIndex.from_product([tutte_azioni, [False, True]], names=['Azione', 'Esito'])     # Create the Cartesian product so each action has both True and False 
        pivot_df = pivot_df.reindex(nuovo_indice, fill_value=0)

        fig = go.Figure()

        # X will be a numeric index to position the bars correctly
        x_indices = list(range(len(pivot_df)))

        # Add one trace for each experiment (pivot column)
        for esperimento in pivot_df.columns:
            fig.add_trace(go.Bar(
                name=esperimento,
                x=x_indices,
                y=pivot_df[esperimento].values,
                hovertext=[f"Azione: {a}<br>Esito: {'S' if e else 'F'}" for a, e in pivot_df.index]
            ))

        # Create labels for the actions (one every two bars)
        tick_vals = [i * 2 + 0.5 for i in range(len(tutte_azioni))]
        tick_text = tutte_azioni

        # Layout configuration
        fig.update_layout(
            barmode='stack',  # Keep experiments stacked on top of each other
            title='Evidence Distribution: Failures (F) vs Successes (S)',
            xaxis=dict(
                tickvals=tick_vals,
                ticktext=tick_text,
                tickangle=-45,  # Slanted action labels
                title="Actions",
                automargin=True
            ),
            yaxis=dict(title='Frequency'),
            legend_title="Experiments",
            template="plotly_white",
            height=700,
            width=1800,
            margin=dict(l=100, r=50, b=150, t=80),
            autosize=False 
        )

        for i, (azione, esito) in enumerate(pivot_df.index):
            fig.add_annotation(
                x=i,
                y=0,
                text="S" if esito else "F",
                showarrow=False,
                yshift=-15,      # Place the letter just below the axis
                font=dict(size=10, color="gray")
            )

        # Separator lines between actions
        for i in range(len(tutte_azioni)):
            fig.add_vline(x=(i * 2) + 1.5, line_width=1, line_dash="dash", line_color="gray", opacity=0.2)

        
        if output_dir:
            try:
                output_path = Path(output_dir)
                output_path = os.path.join(output_path, f"Comparison_Evidence.png")
                fig.write_image(output_path)
            except Exception as e:
                print(f"Warning: Could not save the image due to error: {e}")
                return None
        return fig
        
def plotevidence(analyzer: ResultsAnalyzer, exp: Path, output_dir: Optional[Path] = None):
        # Bar chart for evidence data
        dati_evidence = []          # List used to store evidence data to plot  
        for r in analyzer.results:
            if r.evidence is not None:
                dati_evidence.append(r.evidence)

        if dati_evidence:       # If there is evidence data to plot, generate the bar chart
            df = pd.DataFrame(dati_evidence, columns=['Tupla', 'Condizione'])       # Create a DataFrame with two columns: 'Tupla' for the action and 'Condizione' for success/failure
            df['Etichetta'] = df['Tupla'].apply(lambda x: "-".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x))    # Create the 'Etichetta' column containing the action tuple as a string

            conteggi = pd.crosstab(df['Etichetta'], df['Condizione'])   # Create a contingency table counting how often each action/outcome pair occurs

            cols_mappa = {False: None, True: None} # Create a dictionary for True and False so the conditions can be plotted independently
            for c in conteggi.columns:
                if c in [False, 0, "False", "0"]:   # If the column represents failures, map it to False. The original data type is inconsistent.
                    cols_mappa[False] = c
                elif c in [True, 1, "True", "1"]:
                    cols_mappa[True] = c

            fig = go.Figure()

            # Add Failure trace (red) - only if data exists
            if cols_mappa[False] is not None:
                fig.add_trace(go.Bar(
                    x=conteggi.index.tolist(),
                    y=conteggi[cols_mappa[False]].values,
                    name='Failure',
                    marker_color='red'
                ))

            # Add Success trace (green) - only if data exists
            if cols_mappa[True] is not None:
                fig.add_trace(go.Bar(
                    x=conteggi.index.tolist(),
                    y=conteggi[cols_mappa[True]].values,
                    name='Success',
                    marker_color='green'
                ))

            # Layout configuration to avoid clipped labels
            fig.update_layout(
                title=f'Evidence Distribution: Failures (red) vs Successes (green) - {exp if exp else ""}',
                xaxis_title='Action',
                yaxis_title='Frequency',
                barmode='group',
                template='plotly_white',
                xaxis=dict(
                    type='category',
                    tickmode='array',
                    tickvals=list(range(len(conteggi.index))),
                    ticktext=conteggi.index.tolist(),
                    tickangle=-45,          # Rotate for readability
                    automargin=True,       # Prevent clipped text
                ),
                # Increase the left margin so the first label does not disappear
                margin=dict(l=120, r=50, b=150, t=80),
                height=700,
                width=1600,
                autosize=False
            )

            if output_dir:
                try: 
                    output_path = Path(output_dir)
                    output_path = os.path.join(output_path, f"{exp}_Evidence.png")
                    fig.write_image(output_path)
                except Exception as e:
                    print(f"Warning: Could not save the image due to error: {e}")
                    return None
            return fig

def plottuning_confronto(analyzers: List[ResultsAnalyzer], output_dir: Optional[Path] = None):
    crea_grafico= False
    for analyzer in analyzers:
        if any(r.tuning is not None for r in analyzer.results):
            crea_grafico = True

    if not crea_grafico:
        return None  # If there is no tuning data to plot, exit without creating a chart
    
    tuning_data = []      # List used to store diagnoses found in diagnosis_symbolic_logs
    for analyzer in analyzers:
        exp_name = analyzer.experiment_dir.name 
        for r in analyzer.results:
            if r.tuning is not None:
                for tuning in r.tuning:
                    tuning_data.append({
                        'Tuning': tuning,
                        'Esperimento': exp_name
                    })

    df = pd.DataFrame(tuning_data)
    df_tuning = df.groupby(['Tuning', 'Esperimento']).size().unstack(fill_value=0)

    fig=go.Figure()
    for esperimento in df_tuning.columns:
        fig.add_trace(go.Bar(name=esperimento, x=df_tuning.index.tolist(), y=df_tuning[esperimento].values, hovertext=[f"Tuning: {t}" for t in df_tuning.index]))

    fig.update_layout(
        title='Frequency of Tuning (tuning_symbolic_logs) by Experiment',
        xaxis_title='Tuning',
        yaxis_title='Frequency',
        barmode='stack',
        template='plotly_white',
        xaxis=dict(
            type='category',
            tickmode='array',
            tickvals=list(range(len(df_tuning.index))),     # Numeric positions for the labels
            ticktext=df_tuning.index.tolist(),
            tickangle=-45,          # Rotate for readability
            automargin=True,       # Prevent clipped text
        ),
        margin=dict(l=120, r=50, b=150, t=80),
        height=700,
        width=1800,
        autosize=False
    )
    if output_dir:
        try:
            output_path = Path(output_dir)
            output_path = os.path.join(output_path, f"Comparison_Tuning.png")
            fig.write_image(output_path)
        except Exception as e:
            print(f"Warning: Could not save the image due to error: {e}")
            return None
    return fig

def plottuning(analyzer, exp, output_dir=None):
    # First check whether there is tuning data to plot; if not, exit without creating a chart
    grafico_t = any(r.tuning is not None for r in analyzer.results)
    grafico_evidence = any(r.evidence is not None for r in analyzer.results)
    num_subplot = sum([grafico_t, grafico_evidence])

    if num_subplot == 0:
        return None

    # Create the subplot figure
    fig = make_subplots(
        rows=1, cols=num_subplot, 
        subplot_titles=((['Frequency of Tuning Taken'] if grafico_evidence else []) + (['Frequency of Tuning Found'] if grafico_t else []))
    )

    idx = 1 # Start with the first plot

    # Evidence-file logic (blue chart)
    if grafico_evidence:
        dati_evidence = [r.evidence for r in analyzer.results if r.evidence is not None]        # Extract data
        df = pd.DataFrame(dati_evidence, columns=['Tupla', 'Condizione'])
        # Extract the action from the tuple (first element)
        df['Metodo'] = df['Tupla'].apply(lambda x: x[0] if isinstance(x, (list, tuple)) else x)
        frequenza_azioni = df['Metodo'].value_counts()

        fig.add_trace(go.Bar( x=frequenza_azioni.index.tolist(),y=frequenza_azioni.values,marker_color='blue',name='Tuning',showlegend=False),row=1, col=idx)
        idx += 1

    # Tuning-file logic (orange chart)
    if grafico_t:
        tuning_data = []                    # Extract data
        for r in analyzer.results:
            if r.tuning is not None:
                tuning_data.extend(r.tuning)
        
        df_tuning = pd.DataFrame(tuning_data, columns=['Tuning'])
        frequenza_tuning = df_tuning['Tuning'].value_counts()

        fig.add_trace(go.Bar(x=frequenza_tuning.index.tolist(),y=frequenza_tuning.values,marker_color='orange',name='Tuning',showlegend=False),row=1, col=idx)

    # Layout configuration and label clipping protection
    fig.update_layout(
        template='plotly_white',
        height=700,
        width=800 * num_subplot,
        margin=dict(l=100, r=50, b=150, t=100), # Generous margins for slanted labels
        autosize=False
    )

    # Apply rotation and alignment to all X axes
    fig.update_xaxes(
        tickangle=-45, 
        automargin=True,
        tickfont=dict(size=12)
    )
    
    fig.update_yaxes(title_text="Frequency")

    # Save
    if output_dir:
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(output_path / f"{exp}_tuning.png"))
        except Exception as e:
            print(f"Warning: Could not save the image due to error: {e}")
            return None
    return fig


def plotdiagnosis_confronto(analyzers: List[ResultsAnalyzer], output_dir: Optional[Path] = None):
    crea_grafico= False
    for analyzer in analyzers:
        if any(r.diagnosis is not None for r in analyzer.results):
            crea_grafico = True

    if not crea_grafico:
        return None  # If there is no diagnosis data to plot, exit without creating a chart
    
    diagnosis_data = []      # List used to store diagnoses found in diagnosis_symbolic_logs
    for analyzer in analyzers:
        exp_name = analyzer.experiment_dir.name 
        for r in analyzer.results:
            if r.diagnosis is not None:
                for diagnosis in r.diagnosis:
                    diagnosis_data.append({
                        'Diagnosi': diagnosis,
                        'Esperimento': exp_name
                    })

    df = pd.DataFrame(diagnosis_data)
    df_diagnosis = df.groupby(['Diagnosi', 'Esperimento']).size().unstack(fill_value=0)
    fig=go.Figure()
    for esperimento in df_diagnosis.columns:
        fig.add_trace(go.Bar(
            name=esperimento,
            x=df_diagnosis.index.tolist(),
            y=df_diagnosis[esperimento].values,
            hovertext=[f"Diagnosis: {d}" for d in df_diagnosis.index]
        ))
    fig.update_layout(
        title='Frequency of Diagnoses (diagnosis_symbolic_logs) by Experiment',
        xaxis_title='Diagnosis',
        yaxis_title='Frequency',
        barmode='stack',
        template='plotly_white',
        xaxis=dict(
            type='category',
            tickmode='array',
            tickvals=list(range(len(df_diagnosis.index))),
            ticktext=df_diagnosis.index.tolist(),
            tickangle=-45,          # Rotate for readability
            automargin=True,       # Prevent clipped text
        ),
        margin=dict(l=120, r=50, b=150, t=80),
        height=700,
        width=1800,
        autosize=False
    )
    if output_dir:
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(output_path / f"Comparison_Diagnosis.png"))
        except Exception as e:
            print(f"Warning: Could not save the image due to error: {e}")
            return None
    return fig

def plotdiagnosis(analyzer, exp, output_dir=None):
    # Preliminary data analysis
    grafico_t = any(r.diagnosis is not None for r in analyzer.results)
    grafico_evidence = any(r.evidence is not None for r in analyzer.results)
    num_subplot = sum([grafico_t, grafico_evidence])

    if num_subplot == 0:
        return None

    # Create the subplot figure
    fig = make_subplots(
        rows=1, cols=num_subplot, 
        subplot_titles=((['Frequency of Diagnoses Considered'] if grafico_evidence else []) + (['Frequency of Diagnoses Found'] if grafico_t else []))
    )

    idx = 1 # Plotly column index (starts at 1)

    # Evidence logic (blue chart)
    if grafico_evidence:
        dati_evidence = [r.evidence for r in analyzer.results if r.evidence is not None]
        df = pd.DataFrame(dati_evidence, columns=['Tupla', 'Condizione'])
        # Extract the diagnosis from the tuple (second element)
        df['Diagnosi'] = df['Tupla'].apply(lambda x: x[1] if isinstance(x, (list, tuple)) else x)
        frequenza_azioni = df['Diagnosi'].value_counts()

        fig.add_trace(go.Bar(x=frequenza_azioni.index.tolist(),y=frequenza_azioni.values,marker_color='blue',name='Diagnosis',showlegend=False),row=1, col=idx)
        idx += 1

    # Tuning logic (orange chart)
    if grafico_t:
        diagnosis_data = []
        for r in analyzer.results:
            if r.diagnosis is not None:
                diagnosis_data.extend(r.diagnosis)
        
        df_diagnosis = pd.DataFrame(diagnosis_data, columns=['Diagnosi'])
        frequenza_diagnosi = df_diagnosis['Diagnosi'].value_counts()

        fig.add_trace(go.Bar(x=frequenza_diagnosi.index.tolist(),y=frequenza_diagnosi.values,marker_color='orange',name='Diagnosis', showlegend=False),row=1, col=idx)

    # Layout configuration and label clipping protection
    fig.update_layout(
        template='plotly_white',
        height=700,
        width=800 * num_subplot,
        margin=dict(l=100, r=50, b=150, t=100), # Generous margins for slanted labels
        autosize=False
    )

    # Apply rotation and alignment to all X axes
    fig.update_xaxes(
        tickangle=-45, 
        automargin=True,
        tickfont=dict(size=12)
    )
    
    fig.update_yaxes(title_text="Frequency")

    # Save
    if output_dir:
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(output_path / f"{exp}_diagnosis.png"))
        except Exception as e:
            print(f"Warning: Could not save the image due to error: {e}")
            return None
    return fig

def plottimeline_confronto(analyzers, output_dir=None):
    timeline_data = []
    
    for analyzer in analyzers:
        # Retrieve the experiment name if available for the hover detail
        exp_name = getattr(analyzer, 'name', 'Unknown')
        for r in analyzer.results:
            if r.diagnosis is not None:
                for diag in r.diagnosis:
                    timeline_data.append({'Iteration': r.iteration, 'Evento': diag, 'Tipo': 'Diagnosis', 'Exp': exp_name})
            if r.tuning is not None:
                for tune in r.tuning:
                    timeline_data.append({'Iteration': r.iteration, 'Evento': tune, 'Tipo': 'Tuning', 'Exp': exp_name})

    if not timeline_data:
        return None

    df = pd.DataFrame(timeline_data)
    # Group by to count the frequency (point size)
    df_counts = df.groupby(['Iteration', 'Evento', 'Tipo']).size().reset_index(name='Frequenza')

    fig = go.Figure()

    # Trace for TUNING (red scale)
    tune_df = df_counts[df_counts['Tipo'] == 'Tuning']
    if not tune_df.empty:
        fig.add_trace(go.Scatter(
            x=tune_df['Iteration'],
            y=tune_df['Evento'],
            mode='markers',
            name='Tuning',
            marker=dict(
                size= 10,
                color=tune_df['Frequenza'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Tune Freq", x=1.15), # Moved right so it does not overlap the first one
                line=dict(width=1, color='black')
            ),
            hovertemplate="<b>%{y}</b><br>Iteration: %{x}<br>Frequency: %{marker.color}<extra></extra>",showlegend=False
        ))

    # Trace for DIAGNOSIS (blue scale)
    diag_df = df_counts[df_counts['Tipo'] == 'Diagnosis']
    if not diag_df.empty:
        fig.add_trace(go.Scatter(
            x=diag_df['Iteration'],
            y=diag_df['Evento'],
            mode='markers',
            name='Diagnosis',
            marker=dict(
                size= 10,
                color=diag_df['Frequenza'],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Diag Freq", x=1.0),
                line=dict(width=1, color='black')
            ),
            hovertemplate="<b>%{y}</b><br>Iteration: %{x}<br>Frequency: %{marker.color}<extra></extra>",showlegend=False
        ))

    # Layout and refinements
    fig.update_layout(
        title='Timeline of Diagnoses and Tuning',
        xaxis=dict(title='Iteration', gridcolor='lightgray'),
        yaxis=dict(title='Event', gridcolor='lightgray', autorange="reversed"), # Reversed to read from top to bottom
        template='plotly_white',
        height=700,
        width=1800,
        margin=dict(l=150, r=200, b=100, t=100), # Space for the colorbars on the right
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        autosize=False
    )

    # Save
    if output_dir:
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(output_path / "Comparison_timeline.png"))
        except Exception as e:
            print(f"Warning: Could not save the image due to error: {e}")
            return None

    return fig

def plottimeline(analyzer, exp, output_dir=None):
    timeline_data_diagnosis = []
    timeline_data_tuning = []

    # Data collection: separate diagnosis and tuning so they can be plotted with different colors
    for r in analyzer.results:
        if r.diagnosis is not None:
            for diag in r.diagnosis:
                timeline_data_diagnosis.append({'Iteration': r.iteration, 'Evento': diag})
        if r.tuning is not None:
            for tune in r.tuning:
                timeline_data_tuning.append({'Iteration': r.iteration, 'Evento': tune})

    if not timeline_data_diagnosis and not timeline_data_tuning:
        return None

    fig = go.Figure()

    # Plot DIAGNOSIS first (blue)
    if timeline_data_diagnosis:
        df_diag = pd.DataFrame(timeline_data_diagnosis)

        fig.add_trace(go.Scatter(x=df_diag['Iteration'],y=df_diag['Evento'], mode='markers',name='Diagnosis',
            marker=dict(color='blue', size=12),hovertemplate="<b>Diagnosis</b><br>Iteration: %{x}<br>Evento: %{y}" ))

    # Then plot TUNING (orange)
    if timeline_data_tuning:
        df_tune = pd.DataFrame(timeline_data_tuning)

        fig.add_trace(go.Scatter(x=df_tune['Iteration'],y=df_tune['Evento'],mode='markers', name='Tuning',
            marker=dict(color='orange', size=10, symbol='diamond'),hovertemplate="<b>Tuning</b><br>Iteration: %{x}<br>Evento: %{y}" ))

    # Layout configuration
    fig.update_layout(
        title=f'Timeline of Diagnoses and Tuning - {exp if exp else ""}',
        xaxis=dict(title='Iteration', gridcolor='lightgray'), # Light grid to make reading easier
        yaxis=dict(title='Event', gridcolor='lightgray', automargin=True), 
        template='plotly_white',
        height=700,
        width=1600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=150, r=50, b=80, t=100), # Large left margin for event names
        autosize=False
    )

    # Save
    if output_dir:
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(output_path / f"{exp}_timeline.png"))
        except Exception as e:
            print(f"Warning: Could not save the image due to error: {e}")
            return None
    return fig

def analyze_all_experiments(parent_dir: Path, output_dir: Path = None):
    # Find all subfolders with algorithm_logs
    experiments = []
    parent_dir = Path(parent_dir)
    for item in parent_dir.iterdir():
        if item.is_dir() and os.path.exists(os.path.join(item, "algorithm_logs")):
            experiments.append(item)
    
    if not experiments:
        return None, None
        
    # Analyze each experiment
    lista_analizer = []
    
    for exp_dir in sorted(experiments):        
        analyzer = ResultsAnalyzer(exp_dir)
        if not analyzer.load_results():
            continue

        lista_analizer.append(analyzer)

    return experiments,lista_analizer