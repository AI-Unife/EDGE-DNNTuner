from graphfun import *
from analyze_out import *
import streamlit as st
import os
import platform
import threading
import tkinter as tk
from tkinter import filedialog

@st.cache_data          # Cache the data so each rerun does not reload everything from scratch
def loading():
    return analyze_all_experiments(st.session_state.cartellaInput, st.session_state.cartellaOutput) # Load the experiments and their analyzers for the selected folder

@st.cache_data          # Cache the data so each rerun does not reload everything from scratch
def loading2():
    return find_out(st.session_state.cartellaInput) # Load the experiments and their analyzers for the selected folder


def _is_macos():
    return platform.system().lower() == "darwin"


def _can_use_tk_dialog():
    # Tk requires the main thread to create NSWindow on Apple platforms.
    if _is_macos():
        return threading.current_thread() is threading.main_thread()
    return True


def _ask_directory_safe():
    if not _can_use_tk_dialog():
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    try:
        root.update()
        return filedialog.askdirectory(master=root)
    finally:
        root.destroy()


def folder_selection_phase():
    st.empty()      # Clear the placeholder to remove any elements from previous phases
    st.title("📈 - ANALYZER Symbolic DNN Tuner")

    st.session_state.uso_tk_dialog = _can_use_tk_dialog()  # Check whether Tkinter dialogs can be used safely

    if not st.session_state.uso_tk_dialog:
        st.warning(
            "On macOS/iOS, the native folder picker can crash outside the main thread. "
            "Enter the path manually in the fields below.",
            icon="⚠️"
        )
    
    # Section for the folder to analyze
    st.subheader("📂 - Folder selector for analysis")   # Subtitle
    cola, colb = st.columns([2,5])         # Create two columns and place the button in the first one to match the layout below
    with cola:
        if st.button('Select folder to analyze ->',use_container_width=True):   # When the button is pressed
            cartella_scelta = _ask_directory_safe()  # Open the file dialog only if it is safely supported
            if cartella_scelta:
                st.session_state.cartellaInput = cartella_scelta    # Save it in session state so it can be reused later
                st.rerun()          # Reload the page

    if not st.session_state.uso_tk_dialog:
        percorso_input = st.text_input(
            "Or enter the analysis folder path manually",
            value=st.session_state.get('cartellaInput', ''),
            key="manual_cartella_input"
        )
        if st.button('Use this path (analysis)'):
            if percorso_input and os.path.isdir(percorso_input):
                st.session_state.cartellaInput = percorso_input
                st.rerun()
            else:
                st.error("Invalid path. Please enter an existing folder.", icon="❌")
            
    if 'cartellaInput' in st.session_state:     # If a folder to analyze exists, show a message with the selected path
        st.info(f"Selected folder: **{st.session_state.cartellaInput}**")

    # Optional selection for the output folder
    st.subheader("📂 - Save folder selector [Optional]")
    col1, col2, col_vuota1 = st.columns([2, 1, 4]) 
    with col1:
        if st.button('Select save folder ->',use_container_width=True):
            cartella_scelta = _ask_directory_safe()
            if cartella_scelta:
                st.session_state.cartellaOutput = cartella_scelta
                st.rerun()
    with col2:
        if st.button('No selection',use_container_width=True):
            st.session_state.cartellaOutput = None
            st.rerun()

    if 'cartellaOutput' in st.session_state:
        st.info(f"Selected save folder: **{st.session_state.cartellaOutput}**")

    if not st.session_state.uso_tk_dialog:
        percorso_output = st.text_input(
            "Or enter the save folder path manually",
            value=st.session_state.get('cartellaOutput') or '',
            key="manual_cartella_output"
        )
        if st.button('Use this path (output)'):
            if percorso_output and os.path.isdir(percorso_output):
                st.session_state.cartellaOutput = percorso_output
                st.rerun()
            else:
                st.error("Invalid path. Please enter an existing folder.", icon="❌")

    # Button to go to the next phase
    col_vuota2, col3= st.columns([6, 1]) 
    with col3:
        if st.button('Continue',use_container_width=True):      # When the Continue button is pressed, move to the next phase if an analysis folder was selected
            if 'cartellaInput' in st.session_state:
                if 'cartellaOutput' not in st.session_state:
                    st.session_state.cartellaOutput = None  # If no output folder was selected, set it to None

                st.session_state.fase = 'caricamento'
                st.rerun()  # Reload the page to move to the next phase
            else:
                st.error("Please select a folder to analyze before continuing.", icon="❌")

def loading_phase():
    st.empty()            
    st.title("⚙️ - Load data and select charts")
    
    if 'stato_bottone_singoli' not in st.session_state:     # Create a state variable to track whether the single-plot button was pressed, so the section stays visible
        st.session_state.stato_bottone_singoli = False
    if 'stato_bottone_confronto' not in st.session_state:
        st.session_state.stato_bottone_confronto = False
    if 'stato_bottone_spazio_ricerca' not in st.session_state:
        st.session_state.stato_bottone_spazio_ricerca = False

    with st.spinner('Loading experiments...'):     # Show a spinner while data loads so the app feels responsive
        lista_esperimenti, lista_analizer = loading ()  # Load experiment data and matching analyzers using the cached loader
        if lista_esperimenti and lista_analizer:            # Store both lists in session state so they can be reused later
            st.session_state.lista_esperimenti = lista_esperimenti
            st.session_state.lista_analizer = lista_analizer

        lista_out_files = loading2()     # Return a list of search spaces, each made of dictionaries plus the experiment name
        if lista_out_files:
            st.session_state.lista_out_files = lista_out_files

    if not lista_esperimenti:       # Show a warning message if no experiments were found
        st.warning("❌ - No experiments were found in the selected folder. Please verify that the folder contains the correct data and try again.")
        
    if 'lista_grafici_singoli' not in st.session_state:
        st.session_state.lista_grafici_singoli = None   # Create a state variable to track per-experiment charts so the selection is preserved
    if 'lista_grafici_confronto' not in st.session_state:
        st.session_state.lista_grafici_confronto = None   # Create a state variable to track comparison charts so the selection is preserved

    if lista_esperimenti:
        st.write(f"Found {len(lista_esperimenti)} experiments in the selected folder: ")      # Show what was found
        st.write("Select experiments to analyze:")
        esp_analizzare=st.multiselect("Experiments to analyze:", [exp.name for exp in lista_esperimenti])        # Ask which experiments to analyze

        col1,col2,col3= st.columns([2, 2, 2]) 
        with col1:
            if esp_analizzare and st.button('Individual charts'):
                st.session_state.stato_bottone_singoli = True
                st.rerun()
        with col2:
            if esp_analizzare and st.button('Comparison charts'):
                st.session_state.stato_bottone_confronto = True
                st.rerun()
        with col3:
            if esp_analizzare and st.button('Search space'):
                st.session_state.stato_bottone_spazio_ricerca = True
                st.rerun()

        if esp_analizzare and st.session_state.stato_bottone_singoli:
            st.subheader("You selected charts for individual experiments.")

            if 'scelta_comune' not in st.session_state:     # Store a shared choice so many experiments can reuse the same chart selection
                st.session_state.scelta_comune = None

            grafici_da_fare_singoli = []        # Track what to plot for each experiment
            opzioni_grafici = [         # Available charts
                "Line plot of Accuracy, Score, and Params", 
                "Bar plot of action effectiveness", 
                "Diagnosis bar plot", 
                "Tuning bar plot", 
                "Tuning timeline scatter plot"]

            for i, exp in enumerate(esp_analizzare):
                # If a shared choice exists, reuse it for all experiments without asking again
                if st.session_state.scelta_comune is not None:
                    grafici = st.session_state.scelta_comune
                    st.write(f"✅ {exp}: Applied shared selection: {', '.join(grafici)}")
                else:
                    # Otherwise ask the user
                    grafici = st.multiselect(       # Open a multiselect menu
                        f"Which charts should I generate for {exp}?",
                        opzioni_grafici,
                        key=f"ms_{exp}" # Important to use a different key for each multiselect so Streamlit can distinguish them
                    )
                    
                    # Show the apply-to-all button only if the user selected something in the first experiment. The others do not have a button
                    if i == 0 and grafici:                    
                        if st.button(f"Apply to all"):
                            st.session_state.scelta_comune = grafici
                            st.rerun()

                grafici_da_fare_singoli.append((exp, grafici))     # Keep track of which charts to make for each experiment
            
            st.session_state.lista_grafici_singoli = grafici_da_fare_singoli   # Save it in state so the information is still available when creating charts

        if esp_analizzare and st.session_state.stato_bottone_confronto:
            st.subheader("You selected comparison charts.")

            opzioni_grafici = [         # Available charts
                "Line plot of Accuracy, Score, and Params", 
                "Bar plot of action effectiveness", 
                "Diagnosis bar plot", 
                "Tuning bar plot", 
                "Tuning timeline scatter plot"]
    
            grafici = st.multiselect(f"Which charts should I generate?", opzioni_grafici)       # Open a multiselect menu
            
            st.session_state.lista_grafici_confronto = grafici

        if st.session_state.stato_bottone_spazio_ricerca:
            st.subheader("You selected search space charts.")

    colno,colok= st.columns([6, 2]) 
    with colok:
        if (st.session_state.lista_grafici_singoli or st.session_state.lista_grafici_confronto or st.session_state.stato_bottone_spazio_ricerca) and st.button('Create charts'):  # Show the button to move forward if there are charts to create

            st.session_state.lista_esperimenti=[exp for exp in lista_esperimenti if exp.name in esp_analizzare]   # Filter the experiment list so only the selected ones remain

            st.session_state.fase = 'analisi'
            st.rerun() 
    with colno:
        if st.button('Cancel'):        # If I realize I made a wrong choice, I can go back and restart from scratch (clear cache and session state)
            st.session_state.fase = 'selezione_cartella'
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.clear()
            st.query_params.clear()
            st.rerun()

@st.fragment
def render_chart(exp, analizer):        
    fig = plotaccuracy(analizer, exp)
    chart_key = f"chart_{exp}"
    
    # Render the chart
    event = st.plotly_chart(fig,on_select="rerun",selection_mode="points", key=chart_key)

    # Check selection
    if event and event.get("selection", {}).get("points"):
        points = event["selection"]["points"][0]
        # Safely retrieve the index
        custom_data = points.get("customdata")
        iter_idx = custom_data[0] if isinstance(custom_data, list) else custom_data
        
        if iter_idx is not None:
            dialog_key = f"{exp}:{iter_idx}"
            if st.session_state.get("trend_details_last_opened") != dialog_key:
                st.session_state.trend_details_last_opened = dialog_key
                show_detail(exp, iter_idx, f"{chart_key}_{iter_idx}")

@st.dialog("📈 Trend details")
def show_detail(esperimento, iter_idx, chart_key):
    st.write(f" Details for iteration {iter_idx} of {esperimento}")
    
    # Data retrieval logic
    found = False
    for esp in st.session_state.lista_out_files:
        if esp.exp_name == esperimento:
            st.plotly_chart(esp.grafici_trend(iter_idx))
            found = True
            break
    
    if not found:
        st.error("Data not found.")

#Pop-up dialog to show the description of each chart when the info button is pressed
@st.dialog("ℹ️ Chart description")
def mostra_descrizione(testo):
    st.write(testo)

info_dati = {
        "Accuracy": "These charts show the accuracy, the score, and the number of parameters for the selected experiment.\n The red line represents accuracy, the blue line shows score, and the green line indicates the number of parameters. These statistics are shown for each iteration to track model performance.\n\n\n If you click on a point in the accuracy chart, you can see details for that iteration.",
        "Action Effectiveness": "This chart shows how effective each action was at improving the model's performance.\n For each problem/action combination, there are two bars: successful applications (green) and ineffective applications (red).",
        "Diagnosis": "These charts show how many times each problem was diagnosed in the selected experiment.\n The chart with blue bars represents the number of times the problem was considered, while the chart with orange bars represents the number of times the problem was actually found.",
        "Tuning": "These charts show how many times each tuning action was applied in the selected experiment.\n The chart with blue bars represents the number of times the tuning action was taken, while the chart with orange bars represents the number of times the tuning action was actually recommended.",
        "Timeline": "This chart shows, for each iteration, which tuning action was applied (orange) and what kind of problem was found (blue).",
        "Accuracy_comparison": "These charts compare accuracy, score, and number of parameters across the selected experiments.\n The first chart shows accuracy, the second shows score, and the third shows the number of parameters. Distinct colors represent different experiments for easy comparison.",
        "Action Effectiveness_comparison": "This chart compares how effective each action was at improving model performance across the selected experiments.\n For each problem/action combination there are two bars: one for successful applications (the bar marked 'S') and one for ineffective ones (the bar marked 'F'). Different colors represent different experiments.",
        "Diagnosis_comparison": "This chart compares how many times each problem was diagnosed across the selected experiments.\n Distinct colors represent different experiments.",
        "Tuning_comparison": "This chart compares how many times each tuning action was applied across the selected experiments.\n Distinct colors represent different experiments.",
        "Timeline_comparison": "This chart compares, for each iteration, which tuning action was applied and what kind of problem was found across the selected experiments.\n Tuning actions are shown using a red gradient, while problems use a blue gradient. Darker colors indicate that more experiments applied that tuning action or detected that problem in that iteration.",
        "Search_Space_Charts": "These charts show the search space of the selected experiment.\n Each chart represents a different feature of the search space and shows how that feature changes across iterations."
    }

# Creation of title and info button
def riga_grafico(titolo, chiave_info, esperimento=None):   # I can pass an optional experiment name to show it in the description of the chart, otherwise it will be generic
    
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        st.markdown(f"##### {titolo}")
    with col2:
        # Usiamo una chiave unica per ogni bottone (fondamentale in Streamlit)
        if st.button("ℹ️", key=f"btn_{chiave_info}_{esperimento}", help="Click for chart description"):
            mostra_descrizione(info_dati[chiave_info])


def analysis_phase():
    st.empty()
    st.title("🔍 - Results analysis")

    if 'salvato_tutto' not in st.session_state:
        st.session_state.salvato_tutto = False

    col1,col2= st.columns([8, 2]) 
    with col1:
        if st.session_state.cartellaOutput and st.button('Save all'):    # If an output folder was selected, show a button to save all charts at once; otherwise hide it
            st.session_state.salvato_tutto = True
            st.rerun()  # Reload the page to move to the next phase
            
    with col2:              # If I do not want to do anything else after reviewing them, I can move to the next phase
        if st.button('Finish'):
            st.session_state.fase = 'fine'
            st.rerun()  # Reload the page to move to the next phase

    if st.session_state.stato_bottone_singoli and st.session_state.salvato_tutto:     # In single-chart mode, save only the selected charts for each experiment
        for i in range(len(st.session_state.lista_grafici_singoli)):
            esperimento = st.session_state.lista_grafici_singoli[i] [0]    # Take each experiment in turn
            grafici = st.session_state.lista_grafici_singoli[i] [1]    # Determine what to plot
            if grafici:
                for grafico in grafici:
                    analizer = None
                    for a in st.session_state.lista_analizer:     # Get the analyzer for that experiment so it can be passed to the chart functions
                        if a.exp_name == esperimento:
                            analizer = a
                            break

                    if grafico == "Line plot of Accuracy, Score, and Params":
                        if plotaccuracy(analizer,esperimento, st.session_state.cartellaOutput) is None:
                            st.warning(f"⚠️ - Could not save {grafico} for {esperimento}.")
                    elif grafico == "Bar plot of action effectiveness":
                        if plotevidence(analizer,esperimento, st.session_state.cartellaOutput) is None:
                            st.warning(f"⚠️ - Could not save {grafico} for {esperimento}.")
                    elif grafico == "Diagnosis bar plot":
                        if plotdiagnosis(analizer,esperimento, st.session_state.cartellaOutput) is None:
                            st.warning(f"⚠️ - Could not save {grafico} for {esperimento}.")
                    elif grafico == "Tuning bar plot":
                        if plottuning(analizer,esperimento, st.session_state.cartellaOutput) is None:
                            st.warning(f"⚠️ - Could not save {grafico} for {esperimento}.")
                    elif grafico == "Tuning timeline scatter plot":
                        if plottimeline(analizer,esperimento, st.session_state.cartellaOutput) is None:
                            st.warning(f"⚠️ - Could not save {grafico} for {esperimento}.")

        st.success("✅ - All individual charts were successfully saved in the selected output folder!")

    if st.session_state.stato_bottone_confronto and st.session_state.salvato_tutto:     # In comparison mode, save all comparison charts.
        # Select only the analyzers for the experiments I want to compare
        nomi=[exp.name for exp in st.session_state.lista_esperimenti]
        analizer_confronto = [analizer for analizer in st.session_state.lista_analizer if analizer.exp_name in nomi]
        for grafico in st.session_state.lista_grafici_confronto:
            if grafico == "Line plot of Accuracy, Score, and Params":
                if plotaccuracy_confronto(analizer_confronto, st.session_state.cartellaOutput) is None:
                    st.warning(f"⚠️ - Could not save comparison {grafico}.")
            elif grafico == "Bar plot of action effectiveness":
                if plotevidence_confronto(analizer_confronto, st.session_state.cartellaOutput) is None:
                    st.warning(f"⚠️ - Could not save comparison {grafico}.")
            elif grafico == "Diagnosis bar plot":
                if plotdiagnosis_confronto(analizer_confronto, st.session_state.cartellaOutput) is None:
                    st.warning(f"⚠️ - Could not save comparison {grafico}.")
            elif grafico == "Tuning bar plot":
                if plottuning_confronto(analizer_confronto, st.session_state.cartellaOutput) is None:
                    st.warning(f"⚠️ - Could not save comparison {grafico}.")
            elif grafico == "Tuning timeline scatter plot":
                if plottimeline_confronto(analizer_confronto, st.session_state.cartellaOutput) is None:
                    st.warning(f"⚠️ - Could not save comparison {grafico}.")

        st.success("✅ - All comparison charts were successfully saved in the selected output folder!")

    if st.session_state.stato_bottone_spazio_ricerca and st.session_state.salvato_tutto:     # In search-space mode, save all search-space charts for the selected experiments.
        for spazio_ricerca in st.session_state.lista_out_files:
            if spazio_ricerca and spazio_ricerca.exp_name in [exp.name for exp in st.session_state.lista_esperimenti]:   # Save only the selected experiments' search-space charts
                if spazio_ricerca.grafici_spazio_ricerca(st.session_state.cartellaOutput) is None:
                    st.warning(f"⚠️ - Could not save search space charts for {spazio_ricerca.exp_name}.")
        st.success("✅ - All search space charts were successfully saved in the selected output folder!")
    
    st.session_state.salvato_tutto = False   # Reset the state variable so going back and returning does not re-save everything unintentionally

    if st.session_state.stato_bottone_singoli:     # In single-chart mode, show charts one by one with their save buttons    
        st.subheader("Charts for individual experiments")
        with st.spinner('Generating charts...'):      # Actual chart generation
            for i in range(len(st.session_state.lista_grafici_singoli)):    # Iterate through the selections 
                esperimento = st.session_state.lista_grafici_singoli[i] [0]    # Take each experiment in turn
                grafici = st.session_state.lista_grafici_singoli[i] [1]    # Determine what to plot
                if grafici:                                     # If there are charts to show
                    st.write(f"Analyzing {esperimento}")            # Analyze
                    for grafico in grafici:
                        analizer = None
                        for a in st.session_state.lista_analizer:     # Get the analyzer for that experiment so it can be passed to the chart functions
                            if a.exp_name == esperimento:
                                analizer = a
                                break

                        if grafico == "Line plot of Accuracy, Score, and Params":       # Series of if/elif branches to decide which chart to generate 
                            riga_grafico(f"{grafico} for {esperimento}", f"Accuracy",esperimento)     # Create a title with an info button that shows the description of the chart when pressed
                            if st.session_state.cartellaOutput and st.button(f"Save {grafico} for {esperimento}", key=f"salva_{i}_accuracy"):         # Create a button to save this specific chart
                                if plotaccuracy(analizer,esperimento, st.session_state.cartellaOutput) is None:     # Recreate the chart and save it this time
                                    st.warning(f"⚠️ - Could not save {grafico} for {esperimento}.")
                                else:
                                    st.success(f"✅ - {grafico} for {esperimento} was saved successfully!")      # Show a message indicating the chart was saved successfully
                            render_chart(esperimento, analizer)     # Call a dedicated function to render the chart and handle point selection and details

                        elif grafico == "Bar plot of action effectiveness":
                            riga_grafico(f"{grafico} for {esperimento}", f"Action Effectiveness",esperimento)     # Create a title with an info button that shows the description of the chart when pressed
                            col1, col2, col3 = st.columns([1, 3, 1])
                            with col2:
                                if st.session_state.cartellaOutput and st.button(f"Save {grafico} for {esperimento}", key=f"salva_{i}_evidence"):
                                    if plotevidence(analizer,esperimento, st.session_state.cartellaOutput) is None:
                                        st.warning(f"⚠️ - Could not save {grafico} for {esperimento}.")
                                    else:
                                        st.success(f"✅ - {grafico} for {esperimento} was saved successfully!")
                                
                                fig=plotevidence(analizer,esperimento, None)
                                if fig is not None:
                                    st.plotly_chart(fig)
                                else:
                                    st.warning(f"⚠️ - Could not display {grafico} for {esperimento}.")

                        elif grafico == "Diagnosis bar plot":
                            riga_grafico(f"{grafico} for {esperimento}", f"Diagnosis",esperimento)
                            if st.session_state.cartellaOutput and st.button(f"Save {grafico} for {esperimento}", key=f"salva_{i}_diagnosis"):
                                if plotdiagnosis(analizer,esperimento, st.session_state.cartellaOutput) is None:
                                    st.warning(f"⚠️ - Could not save {grafico} for {esperimento}.")
                                else:
                                    st.success(f"✅ - {grafico} for {esperimento} was saved successfully!")
                            
                            fig=plotdiagnosis(analizer,esperimento, None)
                            if fig is not None:
                                st.plotly_chart(fig)
                            else:
                                st.warning(f"⚠️ - Could not display {grafico} for {esperimento}.")


                        elif grafico == "Tuning bar plot":
                            riga_grafico(f"{grafico} for {esperimento}", f"Tuning",esperimento)
                            if st.session_state.cartellaOutput and st.button(f"Save {grafico} for {esperimento}", key=f"salva_{i}_tuning"):
                                if plottuning(analizer,esperimento, st.session_state.cartellaOutput) is None:
                                    st.warning(f"⚠️ - Could not save {grafico} for {esperimento}.")
                                else:
                                    st.success(f"✅ - {grafico} for {esperimento} was saved successfully!")
                            
                            fig=plottuning(analizer,esperimento,None)
                            if fig is not None:
                                st.plotly_chart(fig)
                            else:
                                st.warning(f"⚠️ - Could not display {grafico} for {esperimento}.")

                        elif grafico == "Tuning timeline scatter plot":
                            riga_grafico(f"{grafico} for {esperimento}", f"Timeline",esperimento)
                            col1, col2, col3 = st.columns([1, 3, 1])
                            with col2:
                                if st.session_state.cartellaOutput and st.button(f"Save {grafico} for {esperimento}", key=f"salva_{i}_timeline"):
                                    if plottimeline(analizer,esperimento, st.session_state.cartellaOutput) is None:
                                        st.warning(f"⚠️ - Could not save {grafico} for {esperimento}.")
                                    else:
                                        st.success(f"✅ - {grafico} for {esperimento} was saved successfully!")
                                
                                fig=plottimeline(analizer,esperimento,None)
                                if fig is not None:
                                    st.plotly_chart(fig)
                                else:
                                    st.warning(f"⚠️ - Could not display {grafico} for {esperimento}.")
                else:
                    st.write(f"No chart selected for {esperimento}, skipping analysis.")        # Warn that nothing was selected for the experiment and continue

    if st.session_state.stato_bottone_confronto:     # In comparison mode, show comparison charts for all selected experiments
        st.subheader("Comparison charts")
        with st.spinner('Generating comparison charts...'):
            # Select only the analyzers for the experiments I want to compare
            nomi=[exp.name for exp in st.session_state.lista_esperimenti]
            analizer_confronto = [analizer for analizer in st.session_state.lista_analizer if analizer.exp_name in nomi]
            
            for grafico in st.session_state.lista_grafici_confronto:
                if grafico == "Line plot of Accuracy, Score, and Params":
                    riga_grafico(f"{grafico} comparison", f"Accuracy_comparison")     
                    if st.session_state.cartellaOutput and st.button(f"Save {grafico}", key=f"salva_confronto_accuracy"):
                        if plotaccuracy_confronto(analizer_confronto, st.session_state.cartellaOutput) is None:
                            st.warning(f"⚠️ - Could not save comparison {grafico}.")
                        else:
                            st.success(f"✅ - Comparison {grafico} was saved successfully!")
                    fig=plotaccuracy_confronto(analizer_confronto)
                    if fig is not None:
                        st.plotly_chart(fig)
                    else:
                        st.warning(f"⚠️ - Could not display {grafico} for {esperimento}.")

                elif grafico == "Bar plot of action effectiveness":
                    riga_grafico(f"{grafico} comparison", f"Action Effectiveness_comparison")
                    if st.session_state.cartellaOutput and st.button(f"Save {grafico}", key=f"salva_confronto_evidence"):
                        if plotevidence_confronto(analizer_confronto, st.session_state.cartellaOutput) is None:
                            st.warning(f"⚠️ - Could not save comparison {grafico}.")
                        else:
                            st.success(f"✅ - Comparison {grafico} was saved successfully!")
                    fig=plotevidence_confronto(analizer_confronto)
                    if fig is not None:
                        st.plotly_chart(fig)
                    else:
                        st.warning(f"⚠️ - Could not display {grafico} for {esperimento}.")


                elif grafico == "Diagnosis bar plot":
                    riga_grafico(f"{grafico} comparison", f"Diagnosis_comparison")
                    cola,colb, colc = st.columns([1, 8, 1])
                    with colb:
                        if st.session_state.cartellaOutput and st.button(f"Save {grafico}", key=f"salva_confronto_diagnosis"):
                            if plotdiagnosis_confronto(analizer_confronto, st.session_state.cartellaOutput) is None:
                                st.warning(f"⚠️ - Could not save comparison {grafico}.")
                            else:
                                st.success(f"✅ - Comparison {grafico} was saved successfully!")
                        fig=plotdiagnosis_confronto(analizer_confronto)
                        if fig is not None:
                            st.plotly_chart(fig)
                        else:
                            st.warning(f"⚠️ - Could not display {grafico} for {esperimento}.")

                elif grafico == "Tuning bar plot":
                    riga_grafico(f"{grafico} comparison", f"Tuning_comparison")
                    cola,colb, colc = st.columns([1, 8, 1])
                    with colb:
                        if st.session_state.cartellaOutput and st.button(f"Save {grafico}", key=f"salva_confronto_tuning"):
                            if plottuning_confronto(analizer_confronto, st.session_state.cartellaOutput) is None:
                                st.warning(f"⚠️ - Could not save comparison {grafico}.")
                            else:
                                st.success(f"✅ - Comparison {grafico} was saved successfully!")
                        
                        fig=plottuning_confronto(analizer_confronto)
                        if fig is not None:
                            st.plotly_chart(fig)
                        else:
                            st.warning(f"⚠️ - Could not display {grafico} for {esperimento}.")

                elif grafico == "Tuning timeline scatter plot":
                    riga_grafico(f"{grafico} comparison", f"Timeline_comparison")
                    col4, col2, col3 = st.columns([1, 8, 1])
                    with col2:
                        if st.session_state.cartellaOutput and st.button(f"Save {grafico}", key=f"salva_confronto_timeline"):
                            if plottimeline_confronto(analizer_confronto, st.session_state.cartellaOutput) is None:
                                st.warning(f"⚠️ - Could not save comparison {grafico}.")
                            else:
                                st.success(f"✅ - Comparison {grafico} was saved successfully!")
                        
                        fig=plottimeline_confronto(analizer_confronto)
                        if fig is not None:
                            st.plotly_chart(fig)
                        else:
                            st.warning(f"⚠️ - Could not display {grafico} for {esperimento}.")

    if st.session_state.stato_bottone_spazio_ricerca:     # In search-space mode, show search-space charts for all selected experiments
        st.subheader("Search space charts")
        with st.spinner('Generating search space charts...'):
            for esperimento in st.session_state.lista_out_files:
                if esperimento and esperimento.exp_name in [exp.name for exp in st.session_state.lista_esperimenti] and st.session_state.cartellaOutput and st.button(f"Save search space charts for {esperimento.exp_name}", key=f"salva_spazio_ricerca_{esperimento.exp_name}"):
                    if esperimento.grafici_spazio_ricerca(st.session_state.cartellaOutput) is None:
                        st.warning(f"⚠️ - Could not save search space charts for {esperimento.exp_name}.")
                    else:
                        st.success(f"✅ - Search space charts for {esperimento.exp_name} was saved successfully!")
                if esperimento and esperimento.exp_name in [exp.name for exp in st.session_state.lista_esperimenti]:   # In search-space mode, show the selected experiments' search-space charts.
                    riga_grafico(f"Search space charts for {esperimento.exp_name}", f"Search_Space_Charts", esperimento.exp_name)
                    fig=esperimento.grafici_spazio_ricerca()
                    col1, col2, col3 = st.columns([1, 8, 1])
                    with col2:
                        if fig is not None:
                            st.plotly_chart(fig)
                        else:
                            st.warning(f"⚠️ - Could not display {grafico} for {esperimento}.")
        

def main():
    # Set the app name, icon, and layout (how the page is structured)
    st.set_page_config(page_title="ANALYZER Symbolic DNN Tuner", page_icon="📈", layout="wide")
    
    if 'fase' not in st.session_state:                      # Create a state variable to manage the app phases
        st.session_state.fase = 'selezione_cartella'        # Set the initial phase: folder selection
    if 'trend_details_last_opened' not in st.session_state:
        st.session_state.trend_details_last_opened = None

    placeholder = st.empty()        # Create a placeholder to dynamically manage page content based on the current phase
    with placeholder.container():
        
        # Phase 1: Folder selection
        if st.session_state.fase == 'selezione_cartella':
            folder_selection_phase()
            
        # Phase 2: Data loading
        elif st.session_state.fase == 'caricamento':
            loading_phase()            

        # Phase 3: Analysis
        elif st.session_state.fase == 'analisi':
            analysis_phase()

        # Final phase
        elif st.session_state.fase == 'fine':
            st.empty()
            st.title("✅ - Analysis completed!")
            st.write("The results analysis was completed successfully. ")
            st.balloons()

            col_vuota, col= st.columns([5, 2]) 
            with col:                      # If I want a new analysis, I can go back to the initial phase and clear all state
                if st.button('Back to the initial phase',use_container_width=True):
                    st.session_state.fase = 'selezione_cartella'
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.session_state.clear()
                    st.query_params.clear()
                    st.rerun()

if __name__ == "__main__":
    main()