from __future__ import annotations

import os
import re
from typing import Dict, List, Iterable
import random
import numpy as np

from problog.program import PrologString
from problog import get_evaluatable

from exp_config import load_cfg


class NeuralSymbolicBridge:
    """
    Bridge between numeric training signals and the ProbLog symbolic layer.
    Responsibilities:
      - Build a complete ProbLog program by stitching facts + problems + base model + new rules.
      - Update learned probabilities inside the current probabilistic rule set.
      - Run inference to obtain action probabilities conditioned on observed facts.
    """

    def __init__(self):
        """
        Initialize the list of fact "functors" and the canonical problem names.
        The order in `initial_facts` must match the order of numeric facts you pass later.
        """
        self.initial_facts: List[str] = [
            "l", "sl", "a", "sa", "vl", "va", "int_loss", "int_slope", "lacc", "hloss", "grad_global_norm", "vanish_th", "exploding_th"
        ]
        self.problems: List[str] = [
            "overfitting", "underfitting", "inc_loss", "floating_loss", "high_lr", "low_lr", "gradient"
        ]
        self.cfg = load_cfg()
    # -------------------------------------------------------------------------
    # Program assembly
    # -------------------------------------------------------------------------

    def build_symbolic_model(self, facts: Iterable, rules: str, only_modules=False) -> PrologString:
        """
        Assemble a ProbLog program from:
          - dynamically injected numeric facts (order must match `self.initial_facts`)
          - previously built probabilistic problems file (sym_prob.pl)
          - a static base model (symbolic_analysis.pl)
          - new rules (string `rules`)

        Args:
            facts: sequence of values to encode as facts, aligned with `self.initial_facts`.
            rules: additional ProbLog rules to append.

        Returns:
            PrologString containing the full model ready for evaluation.
        """
        sym_dir = os.path.join(self.cfg.name, "symbolic")
        os.makedirs(sym_dir, exist_ok=True)

        out_path = os.path.join(sym_dir, "final.pl")
        prob_model_path = os.path.join(sym_dir, "sym_prob.pl")
        try:
            with open(prob_model_path, "r") as p:
                sym_prob = p.read()
        except FileNotFoundError:
            sym_prob = ""
            print(f"[warn] Missing probabilistic model: {prob_model_path}")

        if not only_modules:
            base_model_path = os.path.join(sym_dir, "symbolic_analysis.pl")
            # Load base files (allow empty if missing, but warn via print)
            try:
                with open(base_model_path, "r") as f:
                    sym_model = f.read()
            except FileNotFoundError:
                sym_model = ""
                print(f"[warn] Missing base model: {base_model_path}")
            base_facts = self.initial_facts
        else:
            sym_model="% QUERY ----------------------------------------------------------------------------------------------------------------\nquery(action(_,_))."
            base_facts = [fact for fact in self.initial_facts if fact not in ["l", "sl", "a", "sa", "vl", "va", "int_loss", "int_slope", "lacc", "hloss", "grad_global_norm", "vanish_th", "exploding_th"]]

        # Encode numeric facts in the form: functor(value).
        sym_facts_lines = []
        for fa, functor in zip(facts, base_facts):
            sym_facts_lines.append(f"{functor}({fa}).")
        sym_facts = "\n".join(sym_facts_lines)

        full_program = "\n".join([sym_facts, sym_prob, sym_model, rules or ""]).strip() + "\n"

        # Persist a copy for debugging/inspection
        with open(out_path, "w") as output:
            output.write(full_program)

        return PrologString(full_program)

    # -------------------------------------------------------------------------
    # Probability editing
    # -------------------------------------------------------------------------

    def complete_probs(self, sym_model: str, prev_model: str) -> str:
        """
        Replace the probability of each *probabilistic rule* in `prev_model`
        with the (new) probability learned in `sym_model`, preserving rule bodies.

        - We match lines like: 0.7::head(args) :- body.
        - Facts like: 0.3::head(args). are read to collect probabilities, but we only
          overwrite rules in `prev_model`.

        Args:
            sym_model: text containing learned probs (facts of form P::head(...).)
            prev_model: existing prob. rules with bodies to be updated.

        Returns:
            Updated probabilistic model with replaced probabilities where available.
        """
        fact_re = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*::\s*([a-z][A-Za-z0-9_]*(?:\([^)]*\))?)\s*\.\s*$")
        rule_re = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*::\s*([a-z][A-Za-z0-9_]*(?:\([^)]*\))?)\s*:-\s*(.+?)\s*\.\s*$")

        # Map head -> new probability
        prob_map: Dict[str, str] = {}
        for line in sym_model.splitlines():
            m = fact_re.match(line)
            if m:
                p, head = m.group(1), m.group(2)
                prob_map[head] = p

        out_lines: List[str] = []
        for line in prev_model.splitlines():
            line = line.replace(" ", "")
            m = rule_re.match(line)
            if m:
                _, head, body = m.groups()
                p = prob_map.get(head)
                if p is not None:
                    p = np.clip(float(p), 1e-5, 1-1e-5)
                    out_lines.append(f"{p}::{head} :- {body}.")
                    continue
            # pass-through for lines we don't update (facts, comments, or unmatched)
            out_lines.append(line)

        return "\n".join(out_lines)

    def normalize_text(self, s: str) -> str:
        s = re.sub(r"\s+", " ", s.strip())
        s = re.sub(r"\s*,\s*", ", ", s)
        s = re.sub(r"\s*;\s*", " ; ", s)  # if there are explicit ORs
        return s

    def build_sym_prob(self, problems: str) -> None:
        """
        Merge probabilistic rules having the same head into a single rule per head.
        Strategy:
          - load base (sym_prob_base.pl) and append `problems` (string with rules)
          - parse probabilistic rules with bodies (P::head(...) :- body.)
          - for the same head, combine bodies (set-union over comma-separated atoms)
          - average the probabilities across duplicates (keeps numeric signal)
          - re-emit one rule per head with the aggregated body and averaged prob

        Notes:
          - Deterministic atoms / comments are ignored here (kept in base file).
          - This function *overwrites* sym_prob.pl with the merged rules only.
        """
        sym_dir = os.path.join(self.cfg.name, "symbolic")
        os.makedirs(sym_dir, exist_ok=True)

        base_path = os.path.join(sym_dir, "sym_prob_base.pl")
        out_path = os.path.join(sym_dir, "sym_prob.pl")

        try:
            with open(base_path, "r") as f:
                base_model = f.read()
        except FileNotFoundError:
            base_model = ""
            print(f"[warn] Missing sym_prob_base.pl at: {base_path}")

        full_model = base_model + ("" if base_model.endswith("\n") else "\n") + (problems or "")

        heads_to_info: Dict[str, Dict[str, str]] = {}
        other_lines: List[str] = []  # kept if you later want to re-append them

        prob_with_body_re = re.compile(
            r"^\s*([0-9]*\.?[0-9]+)\s*::\s*([a-z][A-Za-z0-9_]*(?:\([^()]*\))?)\s*:-\s*(.+?)\s*\.\s*$"
        )
        comment_or_empty_re = re.compile(r"^\s*(%.*)?$")


        for raw_line in full_model.splitlines():
            line = raw_line.rstrip()

            if comment_or_empty_re.match(line):
                continue

            m = prob_with_body_re.match(line)
            if m:
                prob = float(m.group(1))
                head = self.normalize_text(m.group(2))
                body = self.normalize_text(m.group(3))
                if head not in heads_to_info:
                    heads_to_info[head] = {"prob": prob, "body": body}
                else:
                    # average probs over duplicates and union body atoms (set over ", ")
                    prev_prob = heads_to_info[head]["prob"]
                    prev_body = heads_to_info[head]["body"]
                    merged_prob = (prev_prob + prob) / 2.0
                    merged_body = ", ".join(sorted(set((prev_body + ", " + body).split(", "))))
                    heads_to_info[head] = {"prob": merged_prob, "body": merged_body}
                continue

            # Collect non-probabilistic lines if ever needed
            other_lines.append(line)

        out_lines: List[str] = []
        for head, info in heads_to_info.items():
            out_lines.append(f"{info['prob']}::{head} :- {info['body']}.")
        out_lines.extend(other_lines)
        final_text = "\n".join(out_lines).rstrip() + "\n"
        with open(out_path, "w") as f:
            f.write(final_text)

    def edit_probs(self, sym_model: str) -> None:
        """
        Update probabilities in `sym_prob.pl` using the learned probabilities in `sym_model`.
        """
        sym_dir = os.path.join(self.cfg.name, "symbolic")
        prev_path = os.path.join(sym_dir, "sym_prob.pl")

        try:
            with open(prev_path, "r") as f:
                prev_model = f.read()
        except FileNotFoundError:
            prev_model = ""
            print(f"[warn] Missing sym_prob.pl at: {prev_path}")

        new_text = self.complete_probs(sym_model, prev_model)
        with open(prev_path, "w") as f:
            f.write(new_text)

    # -------------------------------------------------------------------------
    # Reasoning
    # -------------------------------------------------------------------------

    def symbolic_reasoning(self, facts, diagnosis_logs, tuning_logs, rules, controller, space):
        """
        Run the symbolic reasoning pipeline:
          1) build the model from facts + problems + base + rules
          2) evaluate it to obtain probabilities for queries (actions)
          3) group actions by problem; pick the max-probability action per problem
          4) return the sequence of proposed tuning actions and corresponding diagnoses

        Args:
            facts: numeric facts in the order of `self.initial_facts`
            diagnosis_logs: opened file handle for logging diagnoses
            tuning_logs: opened file handle for logging actions
            rules: extra rules to append

        Returns:
            (tuning, diagnosis) as lists (duplicates preserved to keep original behavior)
        """
        tuning: List[str] = []
        diagnosis: List[str] = []
        res: Dict[str, Dict[str, float]] = {}
        problems: List[str] = []

        # 1) Build the complete symbolic program
        symbolic_model = self.build_symbolic_model(facts, rules, controller.only_modules)

        # 2) Evaluate (map query term -> probability)
        symbolic_evaluation = get_evaluatable().create_from(symbolic_model).evaluate()

        # 3) Extract problem names from terms like action(ActionName, ProblemName)
        for term in symbolic_evaluation.keys():
            s = str(term)
            # naive parse: take substring between first "," and ")"
            problems.append(s[s.find(",") + 1 : s.find(")")])
        # Unique problems in insertion order
        problems = list(dict.fromkeys(problems))

        # 4) For each problem, map action -> probability
        for prob in problems:
            inner: Dict[str, float] = {}
            for term, prob_val in symbolic_evaluation.items():
                s = str(term)
                if prob in s:
                    action_name = s[s.find("(") + 1 : s.find(",")]
                    inner[action_name] = prob_val
            res[prob] = inner

        # 5) Turn map into ordered action/diagnosis lists
        for prob in res.keys():
            diagnosis.append(prob)
            if res[prob]:
                # Choose argmax action for this problem (with random tie-breaking)

                # --- Regularization, residual, and DA controls ---
                if True in space['reg_l2'][1].categories and "reg_l2" in res[prob]:
                    res[prob]["reg_l2"] = 0
                elif True not in space['reg_l2'][1].categories and "remove_reg_l2" in res[prob]:
                    res[prob]["remove_reg_l2"] = 0

                if True in space['skip_connection'][1].categories and "add_residual" in res[prob]:
                    res[prob]["add_residual"] = 0

                elif True not in space['skip_connection'][1].categories and "remove_residual" in res[prob]:
                    res[prob]["remove_residual"] = 0

                if True in space['data_augmentation'][1].categories and "data_augmentation" in res[prob]:
                    res[prob]["data_augmentation"] = 0

                # --- Architectural constraints ---
                tot_conv = controller.count_new_cv

                if tot_conv > controller.max_conv and "new_conv_block" in res[prob]:
                    res[prob]["new_conv_block"] = 0
                    print("Reached max conv blocks")

                if controller.count_new_cv <= 0 and "dec_conv_block" in res[prob]:
                    res[prob]["dec_conv_block"] = 0
                    print("Reached min conv blocks")

                if controller.layer_x_block > controller.max_layer_x_block and "inc_conv_layers" in res[prob]:
                    res[prob]["inc_conv_layers"] = 0
                    print("Reached max layers per block")

                if controller.layer_x_block <= 2 and "dec_conv_layers" in res[prob]:
                    res[prob]["dec_conv_layers"] = 0
                    print("Reached min layers per block")

                if controller.count_new_fc > controller.max_fc and "new_fc_layers" in res[prob]:
                    res[prob]["new_fc_layers"] = 0
                    print("Reached max fc layers")

                if controller.count_new_fc <= 0 and "dec_fc_layers" in res[prob]:
                    res[prob]["dec_fc_layers"] = 0
                    print("Reached min fc layers")

                # --- Skip if all actions are zero ---
                vals = res[prob]
                if all(v == 0 for v in vals.values()):
                    print(f"Skipping '{prob}' — all actions are zero.")
                    continue

                # --- Select best action ---
                max_val = max(vals.values())
                max_keys = [k for k, v in vals.items() if v == max_val]
                chosen_action = random.choice(max_keys)
                tuning.append(chosen_action)

        # Log unique sets, but return the raw (possibly duplicated) sequences to preserve behavior
        to_log_tuning = list(dict.fromkeys(tuning))
        to_log_diagnosis = list(dict.fromkeys(diagnosis))
        diagnosis_logs.write(str(to_log_diagnosis) + "\n")
        tuning_logs.write(str(to_log_tuning) + "\n")

        return tuning, diagnosis