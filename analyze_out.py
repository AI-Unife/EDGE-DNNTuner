from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import os

class FileOUT:
    """Class used to determine the search space"""

    def __init__(self, exp_dir: Path):
        # Set the folders 
        self.exp_dir = Path(exp_dir)
        # exp_name is the name of the last folder in exp_dir
        self.exp_name=self.exp_dir.name
        # Create a list of empty dictionaries to store the results
        self.spazio_ricerca = []
        self.trend = []
    
    def load_search_space(self) -> bool:
        """Load the search space from a .out file in the experiment folder."""
        # Look in the experiment folder for a file ending in .out
        search_space_file = None
        for i in self.exp_dir.iterdir():
            if i.is_file() and i.suffix == ".out":
                search_space_file = i
                break

        if not search_space_file:
            return False
        
        try:
            with open(search_space_file, "r") as f:
                # Extract the correct sequence of search spaces. One may be computed and then discarded if problems appear after the iteration starts
                dict = {}
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if "--- ITERATION" in line:
                        # Before moving to the next iteration, save the search space found in the previous one 
                        if dict:
                            self.spazio_ricerca.append(dict)
                        # Extract the current iteration
                        line =line.split("--- ITERATION")[1].strip()
                        line = line.split(" ")[0].strip() # Keep only the numeric part before any spaces
                        iteration = int(line)
                        dict={"iteration": iteration}

                    elif line.startswith("Chosen point:"):
                        # Extract the part after "Chosen point:"
                        space_str = line.split("Chosen point:")[1].strip()
                        # Convert the string into a dictionary
                        space_dict = eval(space_str)
                        dict.update(space_dict)
                        # if in dict there is a new_fc_? that is negative, set it to 0
                        for key, value in dict.items():
                            if key.startswith("new_fc_") and isinstance(value, (int, float)) and value < 0:
                                dict[key] = 0

                # After the last iteration, save the search space that was found
                if dict:
                    self.spazio_ricerca.append(dict)
                
            return True
        except Exception as e:
            print(f"⚠ Error loading search space: {e}")
            return False
        
    def stampa_spazio_ricerca(self):
        """Print the search space in a readable form"""
        print("\n🔍 Search Space:")

        for point in self.spazio_ricerca:
            iteration = point.get("iteration", "N/A")
            print(f"Iteration {iteration}:")
            for key, value in point.items():
                if key != "iteration":
                    print(f"  {key}: {value}")
            print("-" * 40)
    
    def grafici_spazio_ricerca(self, output_path: Optional[Path] = None):
        """Create charts to visualize the search space"""
        if not self.spazio_ricerca:
            return
        
        # Convert the list of dictionaries into a DataFrame
        df = pd.DataFrame(self.spazio_ricerca)
        # if i have some colomns with only 0 values, drop them
        df = df.loc[:, (df != 0).any(axis=0)]

        # Create one chart per search parameter showing how it changes over iterations in the same window
        grafici=[c for c in df.columns if c != "iteration"]
        grafici_da_fare = len(grafici)
        num_col=5
        num_rig=grafici_da_fare//num_col + (1 if grafici_da_fare % num_col > 0 else 0)

        fig = make_subplots(rows=num_rig, cols=num_col, subplot_titles=grafici) 

        for i, grafico in enumerate(grafici):
            row = i // num_col + 1
            col = i % num_col + 1
            fig.add_trace(go.Scatter(x=df["iteration"], y=df[grafico], mode='lines+markers', name=grafico,hovertemplate=f"<b>{col}</b><br>Iter: %{{x}}<br>Val: %{{y}}"), row=row, col=col)

        fig.update_layout(height=150*num_rig, width=200*num_col, title_text=f"Search Space Trends - {self.exp_name}", showlegend=False,template="plotly_white")

        if output_path:
            try:
                output_path = Path(output_path)
                output_path.mkdir(parents=True, exist_ok=True)
                output_path = os.path.join(output_path, f"{self.exp_dir.name}_search_space.png")
                fig.write_image(output_path)
            except Exception as e:
                print(f"Warning: Could not save the image due to error: {e}")
                return None
        return fig
    
    def load_trend(self) -> bool:
        """Load the iteration trend from a .out file in the experiment folder."""
        # Look in the experiment folder for a file ending in .out
        search_space_file = None

        for i in self.exp_dir.iterdir():
            if i.is_file() and i.suffix == ".out":
                search_space_file = i
                break

        if not search_space_file:
            return False
        
        try:
            with open(search_space_file, "r") as f:
                dict = {}
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if "--- ITERATION" in line:
                        if dict:
                            self.trend.append(dict)
                        # Extract the current iteration
                        line =line.split("--- ITERATION")[1].strip()
                        line = line.split(" ")[0].strip() # Keep only the numeric part before any spaces
                        iteration = int(line)
                        dict={"iteration": iteration}

                    elif "Epoch" in line and "Loss" in line and "Accuracy" in line:
                        line=line.split()
                        loss=float(line[4][:-1])
                        accuracy=float(line[6][:-1])

                        if "loss" in dict:
                            dict["loss"].append(loss)
                        else:
                            dict["loss"] = [loss]
                        if "accuracy" in dict:
                            dict["accuracy"].append(accuracy)
                        else:
                            dict["accuracy"] = [accuracy]

                    elif "Epoch" in line and "loss" in line and "acc" in line:
                        # Replace commas and equals signs with spaces
                        line=line.replace("=", " ")
                        line=line.split(" ")
                        loss=float(line[3][:-1])
                        accuracy=float(line[7][:-1])
                        if "loss" in dict:
                            dict["loss"].append(loss)
                        else:
                            dict["loss"] = [loss]
                        if "accuracy" in dict:
                            dict["accuracy"].append(accuracy)
                        else:
                            dict["accuracy"] = [accuracy]
                    
                    elif "loss" in line and "accuracy" in line:
                        line=line.split(" ")
                        loss=float(line[5])
                        accuracy=float(line[8])
                        if "loss" in dict:
                            dict["loss"].append(loss)
                        else:
                            dict["loss"] = [loss]
                        if "accuracy" in dict:
                            dict["accuracy"].append(accuracy)
                        else:
                            dict["accuracy"] = [accuracy]

                if dict:
                    self.trend.append(dict)
                
            return True
        except Exception as e:
            print(f"⚠ Error loading trend: {e}")
            return False
        
    def stampa_trend(self ):
        """Print the trend in a readable form"""
        print("\n🔍 Trend:")

        for point in self.trend:
            iteration = point.get("iteration", "N/A")
            print(f"Iteration {iteration}:")
            for key, value in point.items():
                if key != "iteration":
                    print(f"  {key}: {value}")
            print("-" * 40)
    
    def grafici_trend(self, i: int, output_path: Optional[Path] = None):
        """Given an iteration, create charts showing the accuracy and loss trend for that iteration"""

        data=self.trend[i]
        fig=make_subplots(rows=1, cols=2, subplot_titles=["Loss Trend", "Accuracy Trend"])
        fig.add_trace(go.Scatter( y=data["loss"], mode='lines', name="Loss"), row=1, col=1)
        fig.add_trace(go.Scatter( y=data["accuracy"], mode='lines', name="Accuracy"), row=1, col=2)
        fig.update_layout(height=400, width=800, showlegend=False,template="plotly_white")
        
        return fig

def find_out(exp_dir: Path):
    """Find .out files"""
    list_of_out_files = []
    exp_dir = Path(exp_dir)
    for i in exp_dir.iterdir():
        if i.is_dir():
            esperimento = FileOUT(i)

            if not esperimento.load_trend () :
                continue

            if not esperimento.load_search_space():
                continue

            list_of_out_files.append(esperimento)

    return list_of_out_files

def main():
    """Main function"""
    print("\n" + "="*70)
    print("  EXPERIMENT RESULTS ANALYZER - Symbolic DNN Tuner")
    print("="*70 + "\n")
    
    # Check whether arguments were provided
    if len(sys.argv) > 1:
        # Batch mode: use command line arguments
        exp_dir = Path(sys.argv[1])
        if not exp_dir.exists():
            print(f"Folder not found: {exp_dir}")
            return
        
        print(f"Input folder: {exp_dir}")
    else:
        # Interactive mode
        path_input = input("Path of the folder containing the experiments: ").strip()
        if path_input and Path(path_input).exists():
            exp_dir = Path(path_input)
        
        if exp_dir is None:
            print("No folder selected")
            return
        
        print(f"\nSelected folder: {exp_dir}\n")
    
    # Analyze experiments
    list_of_out_files = find_out(exp_dir)

    for esperimento in list_of_out_files:
        print(f"\nAnalyzing experiment: {esperimento.exp_name}")
        esperimento.stampa_trend()
        esperimento.stampa_spazio_ricerca()
        esperimento.grafici_trend(0, output_path=exp_dir)
        esperimento.grafici_spazio_ricerca(output_path=exp_dir)

    print("\n" + "="*70)
    print("  Analysis completed!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()