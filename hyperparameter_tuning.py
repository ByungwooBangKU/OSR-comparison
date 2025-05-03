import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import optuna
# Ensure plotly is installed for visualization: pip install plotly kaleido
# Kaleido is needed for saving static images
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_slice
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import joblib
import argparse # To use Namespace

# --- 수정: Evaluation 함수 타입 임포트 ---
from typing import Callable, Tuple, Dict, Any, Union
# --- ---

class OptunaHyperparameterTuner:
    """
    Optuna 기반 하이퍼파라미터 튜닝을 위한 클래스
    (Class for hyperparameter tuning based on Optuna)
    """
    def __init__(self, method_name, datamodule, args):
        self.method_name = method_name
        self.datamodule = datamodule
        self.args = args # Store the main args namespace
        self.best_params = None
        self.best_trial_results = None # Store metrics from the best trial
        self.best_score = -float('inf') # Default for maximization
        self.metric = args.tuning_metric if hasattr(args, 'tuning_metric') else 'f1_score' # Default to f1_score
        self.n_trials = args.n_trials if hasattr(args, 'n_trials') else 20
        self.study = None
        # --- 삭제: evaluation_func 제거 (objective 함수 내에서 처리 방식 변경) ---
        # self.evaluation_func = None

        # Create directories for results
        self.results_dir = "tuning_results"
        self.plots_dir = os.path.join(self.results_dir, "plots")
        self.studies_dir = os.path.join(self.results_dir, "studies")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.studies_dir, exist_ok=True)
        print(f"[Optuna Tuner] Initialized for method '{self.method_name}', metric '{self.metric}', {self.n_trials} trials.")

    def _create_study(self):
        """Creates or loads an Optuna study object."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Include seen_class_ratio in the study name for better organization
        study_name = f"{self.method_name}_{self.args.dataset}_{self.args.seen_class_ratio}_{timestamp}"
        storage_name = f"sqlite:///{self.studies_dir}/{study_name}.db"

        print(f"[Optuna Tuner] Creating/Loading study: '{study_name}' with storage: '{storage_name}'")
        return optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="maximize", # Assuming higher metric score is better
            load_if_exists=False # Start a new study each time by default
        )

    def _define_search_space(self, trial):
        """
        Defines the hyperparameter search space for each OSR method.
        (Ensure this only defines tunable parameters for the *specific* method)
        """
        params = {}
        print(f"[Optuna Trial {trial.number}] Defining search space for '{self.method_name}'...")

        # --- Threshold OSR ---
        if self.method_name == 'threshold':
             params['param_threshold'] = trial.suggest_float('param_threshold', 0.05, 0.95, step=0.05)
             print(f"  Suggesting param_threshold: [{0.05}, {0.95}]")

        # --- OpenMax OSR ---
        elif self.method_name == 'openmax':
            params['param_openmax_tailsize'] = trial.suggest_int('param_openmax_tailsize', 10, 200, step=10)
            max_alpha = min(20, self.datamodule.num_seen_classes) if self.datamodule.num_seen_classes else 20
            if max_alpha < 1: max_alpha = 1 # Ensure at least 1
            params['param_openmax_alpha'] = trial.suggest_int('param_openmax_alpha', 1, max_alpha, step=1)
            print(f"  Suggesting param_openmax_tailsize: [{10}, {200}]")
            print(f"  Suggesting param_openmax_alpha: [{1}, {max_alpha}] (Max based on num_seen_classes)")

        # --- CROSR OSR ---
        elif self.method_name == 'crosr':
            params['param_crosr_reconstruction_threshold'] = trial.suggest_float('param_crosr_reconstruction_threshold', 0.1, 0.99, step=0.05)
            params['param_crosr_tailsize'] = trial.suggest_int('param_crosr_tailsize', 20, 200, step=10)
            # --- 수정: CROSR 튜닝 시 학습 관련 파라미터 추가 ---
            params['lr'] = trial.suggest_float('lr', 1e-5, 1e-4, log=True) # AE 학습률 튜닝
            params['param_crosr_recon_weight'] = trial.suggest_float('param_crosr_recon_weight', 0.1, 1.0, step=0.1) # 가중치 튜닝
            print(f"  Suggesting param_crosr_reconstruction_threshold: [{0.1}, {0.99}]")
            print(f"  Suggesting param_crosr_tailsize: [{20}, {200}]")
            print(f"  Suggesting lr: [1e-5, 1e-4] (for AE training)")
            print(f"  Suggesting param_crosr_recon_weight: [0.1, 1.0]")
            # --- ---

        # --- DOC OSR ---
        elif self.method_name == 'doc':
            params['param_doc_k'] = trial.suggest_float('param_doc_k', 1.0, 5.0, step=0.25)
            # --- 수정: DOC 튜닝 시 학습 관련 파라미터 추가 ---
            params['lr'] = trial.suggest_float('lr', 1e-5, 1e-4, log=True) # Classifier 학습률 튜닝
            print(f"  Suggesting param_doc_k: [{1.0}, {5.0}]")
            print(f"  Suggesting lr: [1e-5, 1e-4] (for classifier training)")
            # --- ---

        # --- ADB OSR ---
        elif self.method_name == 'adb':
            # --- 평가 시점 파라미터 ---
            params['param_adb_distance'] = trial.suggest_categorical('param_adb_distance', ['cosine', 'euclidean'])
            print(f"  Suggesting param_adb_distance: ['cosine', 'euclidean']")

            # --- 학습 시점 파라미터 (모델 재학습 필요 시 사용) ---
            params['lr'] = trial.suggest_float('lr', 1e-5, 1e-4, log=True) # Backbone LR
            params['lr_adb'] = trial.suggest_float('lr_adb', 1e-4, 5e-3, log=True) # ADB Params LR
            params['param_adb_delta'] = trial.suggest_float('param_adb_delta', 0.05, 0.4, step=0.05)
            params['param_adb_alpha'] = trial.suggest_float('param_adb_alpha', 0.01, 0.5, log=True)
            params['adb_freeze_backbone'] = trial.suggest_categorical('adb_freeze_backbone', [True, False])
            print(f"  Suggesting lr: [1e-5, 1e-4] (Backbone)")
            print(f"  Suggesting lr_adb: [1e-4, 5e-3] (ADB Params)")
            print(f"  Suggesting param_adb_delta: [0.05, 0.4]")
            print(f"  Suggesting param_adb_alpha: [0.01, 0.5]")
            print(f"  Suggesting adb_freeze_backbone: [True, False]")
            # --- ---
        else:
            print(f"  Warning: No specific search space defined for method '{self.method_name}'.")

        return params

    # --- 수정: objective 함수는 이제 모델 재학습 + 평가 함수를 받음 ---
    def _objective_with_retraining(self, trial, model_training_and_evaluation_func):
        """Optuna trial objective function that INCLUDES model retraining."""
        # 1. Get suggested hyperparameters
        params = self._define_search_space(trial)
        # Create a copy of args for this trial, updated with suggested params
        trial_args = argparse.Namespace(**vars(self.args))
        for name, value in params.items():
            setattr(trial_args, name, value)

        param_str = ", ".join([f"{name.replace('param_', '').replace('lr_adb','LR').replace('adb_','')}"
                               f"={value:.4f}" if isinstance(value, float) else f"{name.replace('param_', '').replace('lr_adb','LR').replace('adb_','')}={value}"
                               for name, value in params.items()])
        print(f"\n--- Optuna Trial {trial.number + 1}/{self.n_trials} (Retraining) ---")
        print(f"Method: {self.method_name.upper()}, Params: {param_str}")

        try:
            # This function handles model initialization, training, AND evaluation
            results_dict, score_float = model_training_and_evaluation_func(trial_args)

            if score_float is None or not np.isfinite(score_float):
                print(f"Warning: Invalid score ({score_float}). Failure.")
                valid_metrics = ['accuracy', 'auroc', 'f1_score', 'unknown_detection_rate']
                for metric_name in valid_metrics:
                     metric_value = results_dict.get(metric_name, float('nan'))
                     trial.set_user_attr(metric_name, float(metric_value) if pd.notna(metric_value) else None)
                return -1e9 # Report failure

            # Store metrics
            valid_metrics = ['accuracy', 'auroc', 'f1_score', 'unknown_detection_rate']
            print("Trial Results:")
            for metric_name in valid_metrics:
                metric_value = results_dict.get(metric_name, float('nan'))
                trial.set_user_attr(metric_name, float(metric_value) if pd.notna(metric_value) else None)
                print(f"  {metric_name}: {metric_value:.4f}")

            print(f"--> Trial {trial.number + 1} Score ({self.metric}): {score_float:.4f}")
            return score_float

        except Exception as e:
            print(f"Error during Optuna trial {trial.number + 1} (Retraining): {e}")
            import traceback
            traceback.print_exc()
            # Store NaN/None in user attributes even on error
            valid_metrics = ['accuracy', 'auroc', 'f1_score', 'unknown_detection_rate']
            for metric_name in valid_metrics:
                trial.set_user_attr(metric_name, None)
            return -1e9 # Report failure

    # --- 수정: 모델 재학습 *없는* objective 함수 ---
    def _objective_evaluate_only(self, trial, model_evaluation_func):
        """Optuna trial objective function that ONLY evaluates (no retraining)."""
        # 1. Get suggested hyperparameters specific to the OSR method (not training ones)
        params = self._define_search_space(trial)
        # Create a copy of args for this trial, updated with suggested params
        trial_args = argparse.Namespace(**vars(self.args))
        for name, value in params.items():
            setattr(trial_args, name, value)

        param_str = ", ".join([f"{name.replace('param_', '')}={value:.4f}" if isinstance(value, float) else f"{name.replace('param_', '')}={value}"
                               for name, value in params.items()])
        print(f"\n--- Optuna Trial {trial.number + 1}/{self.n_trials} (Evaluate Only) ---")
        print(f"Method: {self.method_name.upper()}, Params: {param_str}")

        try:
            # This function takes the *fixed* base model and evaluates with trial_args
            # It returns the results dict and the target score
            results_dict, score_float = model_evaluation_func(trial_args)

            if score_float is None or not np.isfinite(score_float):
                print(f"Warning: Invalid score ({score_float}). Failure.")
                valid_metrics = ['accuracy', 'auroc', 'f1_score', 'unknown_detection_rate']
                for metric_name in valid_metrics:
                     metric_value = results_dict.get(metric_name, float('nan'))
                     trial.set_user_attr(metric_name, float(metric_value) if pd.notna(metric_value) else None)
                return -1e9

            # Store metrics
            valid_metrics = ['accuracy', 'auroc', 'f1_score', 'unknown_detection_rate']
            print("Trial Results:")
            for metric_name in valid_metrics:
                metric_value = results_dict.get(metric_name, float('nan'))
                trial.set_user_attr(metric_name, float(metric_value) if pd.notna(metric_value) else None)
                print(f"  {metric_name}: {metric_value:.4f}")

            print(f"--> Trial {trial.number + 1} Score ({self.metric}): {score_float:.4f}")
            return score_float

        except Exception as e:
            print(f"Error during Optuna trial {trial.number + 1} (Evaluate Only): {e}")
            import traceback
            traceback.print_exc()
            # Store NaN/None in user attributes even on error
            valid_metrics = ['accuracy', 'auroc', 'f1_score', 'unknown_detection_rate']
            for metric_name in valid_metrics:
                 trial.set_user_attr(metric_name, None)
            return -1e9 # Report failure

    # --- 수정: tune 함수는 objective 함수를 선택적으로 받음 ---
    def tune(self,
             objective_func: Callable[[optuna.Trial, Any], float],
             evaluation_func: Any):
        """Performs hyperparameter optimization using the provided objective function."""
        # The 'evaluation_func' is the function passed to the objective
        # It might be 'train_and_evaluate_trial' or 'evaluate_trial_no_retraining' logic

        print(f"\n[Hyperparameter Tuning] Starting Optuna for {self.method_name.upper()}")
        print(f"Optimizing metric: {self.metric}")
        print(f"Number of trials: {self.n_trials}")

        self.study = self._create_study()

        try:
            # Pass the evaluation function (which contains the core logic) to the objective
            self.study.optimize(lambda trial: objective_func(trial, evaluation_func),
                                n_trials=self.n_trials,
                                show_progress_bar=True)
        except KeyboardInterrupt: print("\nOptimization stopped by user.")
        except Exception as e: print(f"\nError during optimization: {e}")

        # --- 결과 처리 (기존과 유사) ---
        if not self.study.trials: print("No trials completed."); return {}, {}
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and t.value > -1e8]
        if not completed_trials:
            print("Warning: No trials completed successfully.")
            self.best_params = get_default_best_params(self.method_name)
            self.best_trial_results = {}
            self.best_score = -float('inf')
        else:
            best_trial = self.study.best_trial
            self.best_params = best_trial.params # Only stores params defined in _define_search_space
            self.best_score = best_trial.value
            self.best_trial_results = { m: best_trial.user_attrs.get(m, None) for m in ['accuracy', 'auroc', 'f1_score', 'unknown_detection_rate'] }
            print(f"\n--- Optuna Tuning Finished for {self.method_name.upper()} ---")
            print(f"Best Trial: {best_trial.number + 1}, Best Score ({self.metric}): {self.best_score:.4f}")
            print("Best Parameters:")
            # Filter out internal/training params if not relevant for final run (e.g., LR for threshold)
            params_to_print = {k: v for k, v in self.best_params.items() if k.startswith('param_')}
            if not params_to_print and self.best_params: # Show all if only training params were tuned
                params_to_print = self.best_params
            [print(f"  {k.replace('param_', '')}: {v}") for k, v in params_to_print.items()]
            print("Metrics for Best Trial:")
            [print(f"  {k}: {v if v is not None else 'N/A'}") for k, v in self.best_trial_results.items()]
            self._save_tuning_results()
            self._visualize_tuning_results()
        return self.best_params, self.best_trial_results

    # --- 이하 _save_tuning_results, _visualize_tuning_results, load_best_params, get_default_best_params는 이전과 동일하게 유지 ---
    # ... (이전 코드 붙여넣기) ...

    def _save_tuning_results(self):
        """Saves the tuning results to a JSON file and the study object."""
        if self.study is None or self.best_params is None:
            print("No tuning results to save.")
            return

        # Use seen_class_ratio in filename
        output_file = os.path.join(self.results_dir, f"{self.method_name}_tuning_{self.args.dataset}_{self.args.seen_class_ratio}.json")
        study_file = os.path.join(self.studies_dir, f"{self.method_name}_tuning_{self.args.dataset}_{self.args.seen_class_ratio}.pkl")

        output_data = {
            'method': self.method_name,
            'dataset': self.args.dataset,
            'seen_class_ratio': self.args.seen_class_ratio,
            'tuning_metric': self.metric,
            'n_trials_completed': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'total_trials_requested': self.n_trials,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_trial_metrics': self.best_trial_results,
            'timestamp': datetime.datetime.now().isoformat()
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"Tuning summary saved to: {output_file}")
        except Exception as e:
            print(f"Error saving tuning summary JSON: {e}")

        try:
            joblib.dump(self.study, study_file)
            print(f"Optuna study object saved to: {study_file}")
        except Exception as e:
            print(f"Error saving Optuna study object: {e}")

    def _visualize_tuning_results(self):
        """Visualizes the Optuna tuning results using plotly."""
        if self.study is None or not self.study.trials:
            print("No study data available for visualization.")
            return

        if len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]) < 2:
            print("Need at least 2 completed trials for meaningful visualization.")
            return

        print("Generating Optuna visualization plots...")
        # Use seen_class_ratio in filenames
        base_filename = os.path.join(self.plots_dir, f"{self.method_name}_{self.args.dataset}_{self.args.seen_class_ratio}")

        try:
            # 1. Optimization History
            fig_hist = plot_optimization_history(self.study)
            fig_hist.write_image(f"{base_filename}_history.png")

            # 2. Parameter Importances (requires >1 parameter and enough trials)
            # --- 수정: completed_trials 필터링 추가 ---
            completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if len(self.study.best_params) > 1 and len(completed_trials) >= 5:
                 try:
                     # --- 수정: importance 계산 시 completed_trials 만 사용하도록 시도 (Optuna v3+) ---
                     # fig_imp = plot_param_importances(self.study, target=lambda t: t.state == optuna.trial.TrialState.COMPLETE)
                     # --- Optuna v2 호환 방식 ---
                     fig_imp = plot_param_importances(self.study)
                     fig_imp.write_image(f"{base_filename}_importance.png")
                 except ValueError as e:
                     print(f"  Could not generate importance plot (likely due to incomparable parameters or insufficient completed trials with variance): {e}")
                 except Exception as e:
                     print(f"  Could not generate importance plot: {e}")

            # 3. Slice Plot
            if self.study.best_params: # Check if there are parameters
                 try:
                     fig_slice = plot_slice(self.study)
                     fig_slice.write_image(f"{base_filename}_slice.png")
                 except ValueError as e:
                     print(f"  Could not generate slice plot: {e}") # Can fail if few trials or param space issues

            # 4. Contour Plot (for pairs of parameters)
            if len(self.study.best_params) >= 2:
                param_names = list(self.study.best_params.keys())
                num_contour_plots = 0
                max_contour_plots = 3
                for i in range(len(param_names)):
                     for j in range(i + 1, len(param_names)):
                          if num_contour_plots >= max_contour_plots: break
                          try:
                              fig_contour = plot_contour(self.study, params=[param_names[i], param_names[j]])
                              fig_contour.write_image(f"{base_filename}_contour_{param_names[i]}_{param_names[j]}.png")
                              num_contour_plots += 1
                          except ValueError as e:
                              # This can fail if a parameter has only one value tested etc.
                              print(f"  Could not generate contour plot for {param_names[i]} vs {param_names[j]}: {e}")
                          except Exception as e:
                              print(f"  Error generating contour plot for {param_names[i]} vs {param_names[j]}: {e}")
                     if num_contour_plots >= max_contour_plots: break

            print(f"Optuna visualization plots saved to: {self.plots_dir}")

        except ImportError:
             print("  Plotly or Kaleido not installed. Skipping visualization.")
             print("  Install using: pip install plotly kaleido")
        except Exception as e:
            print(f"Error during Optuna visualization: {e}")
            import traceback
            traceback.print_exc()


def load_best_params(method_name, dataset, seen_class_ratio):
    """Loads previously saved best hyperparameters from a JSON file."""
    file_path = f"tuning_results/{method_name}_tuning_{dataset}_{seen_class_ratio}.json"
    print(f"[Parameter Loading] Checking for previous tuning results: {file_path}")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'best_params' in data:
                print(f"  Found previous results. Loading best parameters.")
                # Filter out non-param keys if they exist
                params_only = {k: v for k, v in data['best_params'].items() if k.startswith('param_') or 'lr' in k or 'freeze' in k}
                return params_only
            else:
                print("  Found previous results file, but 'best_params' key is missing.")
                return None
        except Exception as e:
            print(f"  Error loading parameters from {file_path}: {e}")
            return None
    else:
        print("  No previous tuning file found.")
        return None

def get_default_best_params(method_name):
    """Provides default hyperparameters for each method, based on papers or common practice."""
    print(f"[Parameter Loading] Getting default parameters for method '{method_name}'...")
    defaults = {}
    # Define defaults only for the *method-specific evaluation parameters* here
    # Training related defaults should be handled by argparse
    if method_name == 'threshold':
        defaults = {'param_threshold': 0.5}
    elif method_name == 'openmax':
        defaults = {'param_openmax_tailsize': 50, 'param_openmax_alpha': 10}
    elif method_name == 'crosr':
        defaults = {'param_crosr_reconstruction_threshold': 0.9, 'param_crosr_tailsize': 100}
         # Training params defaults are in argparse: lr, param_crosr_recon_weight
    elif method_name == 'doc':
        defaults = {'param_doc_k': 3.0}
         # Training param default (lr) is in argparse
    elif method_name == 'adb':
        defaults = { 'param_adb_distance': 'cosine'}
         # Training params defaults are in argparse: lr, lr_adb, param_adb_delta, param_adb_alpha, adb_freeze_backbone
    else:
         print(f"  Warning: No defaults defined for method '{method_name}'.")

    print(f"  Defaults for {method_name}: {defaults}")
    return defaults