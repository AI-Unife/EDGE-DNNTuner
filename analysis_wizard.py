"""
Interactive wizard over the tuning-results analysis scripts: robustness_eval.py
(robustness / time-profile), experiment_stats.py (accuracy / HW cost stats +
scatter plot) and experiment_efficiency_stats.py (total time / bandwidth
latency / discard ratio stats).

It never duplicates any script's argument list or defaults -- it introspects
that script's own build_parser() and asks one question per flag, in the same
order the script defines them (data source first, then analysis-specific
options, then filters, then output dir), showing that flag's own help text and
current default. Pressing Enter always keeps the default. Once every flag has
been asked, it shows the equivalent command line and runs that script's
main() directly (same as running it from the shell), unless you decline.

Usage:
    python analysis_wizard.py
"""

import argparse
import importlib
import shlex
import sys

_SCRIPTS = {
    "1": ("Robustness / time-profile (only gesture datasets)", "robustness_eval"),
    "2": ("Accuracy / HW cost stats + scatter plot", "experiment_stats"),
    "3": ("Efficiency stats (total time / bandwidth latency / discard ratio)", "experiment_efficiency_stats"),
}


def _is_bool_flag(action: argparse.Action) -> bool:
    return isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))


def prompt_for_action(action: argparse.Action) -> list:
    """Ask one question for this argparse action; returns the CLI tokens to
    append to argv, or [] to leave it out (i.e. keep the parser's own default)."""
    if isinstance(action, argparse._HelpAction) or not action.option_strings:
        return []

    flag = action.option_strings[0]
    aliases = action.option_strings[1:]
    print(f"\n{flag}" + (f"  (aka {', '.join(aliases)})" if aliases else ""))
    if action.help:
        print(f"  {action.help}")
    if action.choices:
        print(f"  choices: {', '.join(str(c) for c in action.choices)}")

    if _is_bool_flag(action):
        default_str = "y" if action.default else "n"
        raw = input(f"  enable? [y/n, default={default_str}]: ").strip().lower()
        if raw == "":
            return []
        return [flag] if raw in ("y", "yes", "1", "true") else []

    default_display = "none" if action.default is None else str(action.default)
    while True:
        raw = input(f"  value (default: {default_display}; Enter to keep it): ").strip()
        if raw == "":
            return []
        values = raw.split() if action.nargs in ("+", "*") else [raw]
        if action.choices and any(v not in [str(c) for c in action.choices] for v in values):
            print(f"  invalid choice, pick from: {', '.join(str(c) for c in action.choices)}")
            continue
        return [flag] + values


def run_once() -> None:
    print("\nWhich analysis do you want to run?")
    for key, (label, _) in _SCRIPTS.items():
        print(f"  {key}) {label}")
    print("  q) quit")
    choice = input("> ").strip().lower()
    if choice in ("q", "quit", "exit"):
        raise SystemExit(0)
    if choice not in _SCRIPTS:
        print("Not a valid choice.")
        return

    label, module_name = _SCRIPTS[choice]
    module = importlib.import_module(module_name)
    if module.__doc__:
        print(f"\n{'='*70}\n{label}\n{'='*70}")
        print(module.__doc__.strip())

    parser = module.build_parser()
    argv = [f"{module_name}.py"]
    print(f"\n{'-'*70}\nAnswer each question (Enter = keep the shown default).\n{'-'*70}")
    for action in parser._actions:
        argv += prompt_for_action(action)

    print("\nEquivalent command:")
    print("  python3 " + " ".join(shlex.quote(a) for a in argv))
    confirm = input("\nRun it now? [Y/n]: ").strip().lower()
    if confirm not in ("", "y", "yes"):
        print("Skipped.")
        return

    old_argv = sys.argv
    sys.argv = argv
    try:
        module.main()
    except SystemExit as e:
        if e.code not in (0, None):
            print(f"\n[{module_name} exited with code {e.code}]")
    finally:
        sys.argv = old_argv


def main():
    print("="*70)
    print("  Tuning-results analysis wizard")
    print("="*70)
    while True:
        try:
            run_once()
        except SystemExit:
            print("\nBye.")
            return
        again = input("\nRun another analysis? [y/N]: ").strip().lower()
        if again not in ("y", "yes"):
            print("\nBye.")
            return


if __name__ == "__main__":
    main()
