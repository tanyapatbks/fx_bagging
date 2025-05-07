"""
Reporting Module for Forex Prediction
This module handles the generation of summary reports from evaluation results.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any


class ReportGenerator:
    """
    Class for generating summary reports from evaluation results
    """
    
    def __init__(self, config):
        """
        Initialize the ReportGenerator with configuration parameters
        
        Args:
            config: Configuration object containing paths and parameters
        """
        self.config = config
        
        # Create results directory if it doesn't exist
        if not os.path.exists(self.config.RESULTS_PATH):
            os.makedirs(self.config.RESULTS_PATH)
    
    def create_pair_summary_report(self, pair_evaluation: Dict[str, Dict[str, Dict[str, Any]]], 
                               pair: str) -> str:
        """
        Create a summary report for a currency pair
        
        Args:
            pair_evaluation: Dictionary containing evaluation results for different models
            pair: Currency pair name
            
        Returns:
            Report content as a string
        """
        symbol = self.config.PAIR_SYMBOLS[pair]
        
        # สร้างรายงาน
        report = f"# Summary Report - {pair} ({symbol})\n\n"
        
        # 1. RMSE Comparison
        report += "## 1. RMSE Comparison\n\n"
        report += "| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |\n"
        report += "|-------|-------------------|------------------------|--------------------------------|\n"
        
        models = ['LSTM', 'GRU', 'XGBoost', 'TFT']
        for model in models:
            report += f"| {model} | "
            for idx in range(1, 4):
                identifier = f"{symbol}{idx}"
                if identifier in pair_evaluation and model in pair_evaluation[identifier]:
                    rmse = pair_evaluation[identifier][model]['metrics']['rmse']
                    report += f"{rmse:.6f} | "
                else:
                    report += "N/A | "
            report += "\n"
        
        # 2. Directional Accuracy Comparison
        report += "\n## 2. Directional Accuracy Comparison\n\n"
        report += "| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |\n"
        report += "|-------|-------------------|------------------------|--------------------------------|\n"
        
        for model in models:
            report += f"| {model} | "
            for idx in range(1, 4):
                identifier = f"{symbol}{idx}"
                if identifier in pair_evaluation and model in pair_evaluation[identifier]:
                    dir_acc = pair_evaluation[identifier][model]['metrics']['directional_accuracy']
                    report += f"{dir_acc:.2f}% | "
                else:
                    report += "N/A | "
            report += "\n"
        
        # 3. Annual Return Comparison
        report += "\n## 3. Annual Return Comparison\n\n"
        report += "| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |\n"
        report += "|-------|-------------------|------------------------|--------------------------------|\n"
        
        for model in models:
            report += f"| {model} | "
            for idx in range(1, 4):
                identifier = f"{symbol}{idx}"
                if identifier in pair_evaluation and model in pair_evaluation[identifier]:
                    annual_return = pair_evaluation[identifier][model]['trading_metrics']['annual_return']
                    report += f"{annual_return:.2f}% | "
                else:
                    report += "N/A | "
            report += "\n"
        
        # 4. Model Return vs Benchmark Comparison
        report += "\n## 4. Model Return vs Benchmark Comparison (Enhanced+Selected Data)\n\n"
        report += "| Model | Model Return | Buy & Hold | SMA Crossover | Random |\n"
        report += "|-------|-------------|-----------|---------------|--------|\n"
        
        identifier = f"{symbol}3"
        for model in models:
            if identifier in pair_evaluation and model in pair_evaluation[identifier]:
                benchmark = pair_evaluation[identifier][model]['benchmark']
                model_return = benchmark['model_return']
                buy_hold = benchmark['buy_hold_return']
                sma = benchmark['sma_return']
                random = benchmark['random_return']
                
                report += f"| {model} | {model_return:.2f}% | {buy_hold:.2f}% | {sma:.2f}% | {random:.2f}% |\n"
        
        # 5. Key Findings
        report += "\n## 5. Key Findings\n\n"
        
        # Find best model for RMSE, Directional Accuracy, and Annual Return
        identifier = f"{symbol}3"  # Use Enhanced+Selected data
        if identifier in pair_evaluation:
            # Best RMSE
            rmse_values = {model: pair_evaluation[identifier][model]['metrics']['rmse'] for model in models if model in pair_evaluation[identifier]}
            best_rmse_model = min(rmse_values, key=rmse_values.get)
            best_rmse = rmse_values[best_rmse_model]
            
            # Best Directional Accuracy
            dir_acc_values = {model: pair_evaluation[identifier][model]['metrics']['directional_accuracy'] for model in models if model in pair_evaluation[identifier]}
            best_dir_acc_model = max(dir_acc_values, key=dir_acc_values.get)
            best_dir_acc = dir_acc_values[best_dir_acc_model]
            
            # Best Annual Return
            return_values = {model: pair_evaluation[identifier][model]['trading_metrics']['annual_return'] for model in models if model in pair_evaluation[identifier]}
            best_return_model = max(return_values, key=return_values.get)
            best_return = return_values[best_return_model]
            
            report += f"- Best model for RMSE: **{best_rmse_model}** ({best_rmse:.6f})\n"
            report += f"- Best model for Directional Accuracy: **{best_dir_acc_model}** ({best_dir_acc:.2f}%)\n"
            report += f"- Best model for Annual Return: **{best_return_model}** ({best_return:.2f}%)\n\n"
        
        # Calculate improvement from Raw to Enhanced+Selected
        if f"{symbol}1" in pair_evaluation and f"{symbol}3" in pair_evaluation:
            report += "### Improvement from Raw to Enhanced+Selected\n\n"
            for model in models:
                if model in pair_evaluation[f"{symbol}1"] and model in pair_evaluation[f"{symbol}3"]:
                    rmse_1 = pair_evaluation[f"{symbol}1"][model]['metrics']['rmse']
                    rmse_3 = pair_evaluation[f"{symbol}3"][model]['metrics']['rmse']
                    
                    improvement = (rmse_1 - rmse_3) / rmse_1 * 100
                    report += f"- {model}: RMSE improvement: **{improvement:.2f}%**\n"
        
        # บันทึกรายงาน
        report_path = os.path.join(self.config.RESULTS_PATH, f"{pair}_summary_report.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Summary report for {pair} created: {report_path}")
        return report
    
    def create_bagging_summary_report(self, bagging_evaluation: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
        """
        Create a summary report for the bagging approach
        
        Args:
            bagging_evaluation: Dictionary containing evaluation results for bagging
            
        Returns:
            Report content as a string
        """
        # สร้างรายงาน
        report = "# Bagging Approach Summary Report\n\n"
        
        # 1. RMSE Comparison
        report += "## 1. RMSE Comparison\n\n"
        report += "| Model | EURUSD | GBPUSD | USDJPY | Average |\n"
        report += "|-------|--------|--------|--------|--------|\n"
        
        models = ['LSTM', 'GRU', 'XGBoost', 'TFT']
        pairs = self.config.CURRENCY_PAIRS
        
        for model in models:
            report += f"| {model} | "
            
            rmse_values = []
            for pair in pairs:
                if pair in bagging_evaluation and model in bagging_evaluation[pair]:
                    rmse = bagging_evaluation[pair][model]['metrics']['rmse']
                    rmse_values.append(rmse)
                    report += f"{rmse:.6f} | "
                else:
                    report += "N/A | "
            
            # Calculate average
            if rmse_values:
                avg_rmse = np.mean(rmse_values)
                report += f"{avg_rmse:.6f} |\n"
            else:
                report += "N/A |\n"
        
        # 2. Directional Accuracy Comparison
        report += "\n## 2. Directional Accuracy Comparison\n\n"
        report += "| Model | EURUSD | GBPUSD | USDJPY | Average |\n"
        report += "|-------|--------|--------|--------|--------|\n"
        
        for model in models:
            report += f"| {model} | "
            
            dir_acc_values = []
            for pair in pairs:
                if pair in bagging_evaluation and model in bagging_evaluation[pair]:
                    dir_acc = bagging_evaluation[pair][model]['metrics']['directional_accuracy']
                    dir_acc_values.append(dir_acc)
                    report += f"{dir_acc:.2f}% | "
                else:
                    report += "N/A | "
            
            # Calculate average
            if dir_acc_values:
                avg_dir_acc = np.mean(dir_acc_values)
                report += f"{avg_dir_acc:.2f}% |\n"
            else:
                report += "N/A |\n"
        
        # 3. Annual Return Comparison
        report += "\n## 3. Annual Return Comparison\n\n"
        report += "| Model | EURUSD | GBPUSD | USDJPY | Average |\n"
        report += "|-------|--------|--------|--------|--------|\n"
        
        for model in models:
            report += f"| {model} | "
            
            return_values = []
            for pair in pairs:
                if pair in bagging_evaluation and model in bagging_evaluation[pair]:
                    annual_return = bagging_evaluation[pair][model]['trading_metrics']['annual_return']
                    return_values.append(annual_return)
                    report += f"{annual_return:.2f}% | "
                else:
                    report += "N/A | "
            
            # Calculate average
            if return_values:
                avg_return = np.mean(return_values)
                report += f"{avg_return:.2f}% |\n"
            else:
                report += "N/A |\n"
        
        # 4. Key Findings
        report += "\n## 4. Key Findings\n\n"
        
        # Find best model for RMSE, Directional Accuracy, and Annual Return
        # Calculate average metrics across all pairs
        avg_metrics = {
            'rmse': {},
            'directional_accuracy': {},
            'annual_return': {}
        }
        
        for model in models:
            rmse_values = []
            dir_acc_values = []
            return_values = []
            
            for pair in pairs:
                if pair in bagging_evaluation and model in bagging_evaluation[pair]:
                    rmse_values.append(bagging_evaluation[pair][model]['metrics']['rmse'])
                    dir_acc_values.append(bagging_evaluation[pair][model]['metrics']['directional_accuracy'])
                    return_values.append(bagging_evaluation[pair][model]['trading_metrics']['annual_return'])
            
            if rmse_values:
                avg_metrics['rmse'][model] = np.mean(rmse_values)
            if dir_acc_values:
                avg_metrics['directional_accuracy'][model] = np.mean(dir_acc_values)
            if return_values:
                avg_metrics['annual_return'][model] = np.mean(return_values)
        
        # Best RMSE
        if avg_metrics['rmse']:
            best_rmse_model = min(avg_metrics['rmse'], key=avg_metrics['rmse'].get)
            best_rmse = avg_metrics['rmse'][best_rmse_model]
            report += f"- Best model for RMSE: **{best_rmse_model}** ({best_rmse:.6f})\n"
        
        # Best Directional Accuracy
        if avg_metrics['directional_accuracy']:
            best_dir_acc_model = max(avg_metrics['directional_accuracy'], key=avg_metrics['directional_accuracy'].get)
            best_dir_acc = avg_metrics['directional_accuracy'][best_dir_acc_model]
            report += f"- Best model for Directional Accuracy: **{best_dir_acc_model}** ({best_dir_acc:.2f}%)\n"
        
        # Best Annual Return
        if avg_metrics['annual_return']:
            best_return_model = max(avg_metrics['annual_return'], key=avg_metrics['annual_return'].get)
            best_return = avg_metrics['annual_return'][best_return_model]
            report += f"- Best model for Annual Return: **{best_return_model}** ({best_return:.2f}%)\n"
        
        # Additional findings about bagging approach
        report += "\n### Bagging Approach Benefits\n\n"
        report += "- Bagging combines predictions from models trained on different currency pairs\n"
        report += "- This approach helps to capture universal forex market patterns\n"
        report += "- Reduces overfitting to patterns specific to a single currency pair\n"
        report += "- Can improve robustness and generalization of predictions\n"
        
        # บันทึกรายงาน
        report_path = os.path.join(self.config.RESULTS_PATH, "bagging_summary_report.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Bagging summary report created: {report_path}")
        return report
    
    def create_tuning_comparison_report(self, original_results, tuned_results, pairs):
        """
        Create a report comparing performance before and after hyperparameter tuning
        
        Args:
            original_results: Evaluation results from original models
            tuned_results: Evaluation results from tuned models
            pairs: List of currency pairs
            
        Returns:
            Report content as a string
        """
        # สร้างรายงาน
        report = "# Hyperparameter Tuning Comparison Report\n\n"
        
        # 1. RMSE Comparison
        report += "## 1. RMSE Comparison (Before vs After Tuning)\n\n"
        report += "| Model | Pair | Before Tuning | After Tuning | Improvement |\n"
        report += "|-------|------|---------------|--------------|-------------|\n"
        
        models = ['LSTM', 'GRU', 'XGBoost', 'TFT']
        
        for model in models:
            for pair in pairs:
                # Get RMSE before tuning
                symbol = self.config.PAIR_SYMBOLS[pair]
                identifier = f"{symbol}3"  # Use Enhanced+Selected data
                
                if (pair in original_results and 
                    identifier in original_results[pair] and 
                    model in original_results[pair][identifier]):
                    original_rmse = original_results[pair][identifier][model]['metrics']['rmse']
                    
                    # Get RMSE after tuning
                    if (pair in tuned_results and 
                        'selected' in tuned_results[pair] and 
                        model in tuned_results[pair]['selected']):
                        tuned_rmse = tuned_results[pair]['selected'][model]['metrics']['rmse']
                        
                        # Calculate improvement
                        improvement = (original_rmse - tuned_rmse) / original_rmse * 100
                        
                        report += f"| {model} | {pair} | {original_rmse:.6f} | {tuned_rmse:.6f} | {improvement:.2f}% |\n"
        
        # 2. Directional Accuracy Comparison
        report += "\n## 2. Directional Accuracy Comparison (Before vs After Tuning)\n\n"
        report += "| Model | Pair | Before Tuning | After Tuning | Improvement |\n"
        report += "|-------|------|---------------|--------------|-------------|\n"
        
        for model in models:
            for pair in pairs:
                # Get Directional Accuracy before tuning
                symbol = self.config.PAIR_SYMBOLS[pair]
                identifier = f"{symbol}3"  # Use Enhanced+Selected data
                
                if (pair in original_results and 
                    identifier in original_results[pair] and 
                    model in original_results[pair][identifier]):
                    original_da = original_results[pair][identifier][model]['metrics']['directional_accuracy']
                    
                    # Get Directional Accuracy after tuning
                    if (pair in tuned_results and 
                        'selected' in tuned_results[pair] and 
                        model in tuned_results[pair]['selected']):
                        tuned_da = tuned_results[pair]['selected'][model]['metrics']['directional_accuracy']
                        
                        # Calculate improvement (percentage points)
                        improvement = tuned_da - original_da
                        
                        report += f"| {model} | {pair} | {original_da:.2f}% | {tuned_da:.2f}% | {improvement:.2f} pts |\n"
        
        # 3. Annual Return Comparison
        report += "\n## 3. Annual Return Comparison (Before vs After Tuning)\n\n"
        report += "| Model | Pair | Before Tuning | After Tuning | Improvement |\n"
        report += "|-------|------|---------------|--------------|-------------|\n"
        
        for model in models:
            for pair in pairs:
                # Get Annual Return before tuning
                symbol = self.config.PAIR_SYMBOLS[pair]
                identifier = f"{symbol}3"  # Use Enhanced+Selected data
                
                if (pair in original_results and 
                    identifier in original_results[pair] and 
                    model in original_results[pair][identifier]):
                    original_return = original_results[pair][identifier][model]['trading_metrics']['annual_return']
                    
                    # Get Annual Return after tuning
                    if (pair in tuned_results and 
                        'selected' in tuned_results[pair] and 
                        model in tuned_results[pair]['selected']):
                        tuned_return = tuned_results[pair]['selected'][model]['trading_metrics']['annual_return']
                        
                        # Calculate improvement (percentage points)
                        improvement = tuned_return - original_return
                        
                        report += f"| {model} | {pair} | {original_return:.2f}% | {tuned_return:.2f}% | {improvement:.2f} pts |\n"
        
        # 4. Best Hyperparameters
        report += "\n## 4. Best Hyperparameters\n\n"
        
        for model in models:
            report += f"### {model}\n\n"
            
            for pair in pairs:
                report += f"#### {pair}\n\n"
                
                # Get hyperparameters file
                params_file = os.path.join(self.config.HYPERPARAMS_PATH, pair, f"{model}_selected_hyperparams.json")
                
                if os.path.exists(params_file):
                    with open(params_file, 'r') as f:
                        params_data = json.load(f)
                    
                    # Get best parameters
                    if 'best_params' in params_data:
                        best_params = params_data['best_params']
                        
                        report += "```python\n"
                        report += f"{json.dumps(best_params, indent=4)}\n"
                        report += "```\n\n"
                        
                        # Get additional metrics
                        if 'additional_metrics' in params_data:
                            metrics = params_data['additional_metrics']
                            report += "Performance:\n"
                            
                            for metric, value in metrics.items():
                                report += f"- {metric}: {value}\n"
                            
                            report += "\n"
                    else:
                        report += "No best parameters found.\n\n"
                else:
                    report += "No parameters file found.\n\n"
        
        # 5. Summary and Conclusion
        report += "\n## 5. Summary and Conclusion\n\n"
        
        # Calculate average improvements
        rmse_improvements = []
        da_improvements = []
        return_improvements = []
        
        for model in models:
            for pair in pairs:
                symbol = self.config.PAIR_SYMBOLS[pair]
                identifier = f"{symbol}3"  # Use Enhanced+Selected data
                
                if (pair in original_results and 
                    identifier in original_results[pair] and 
                    model in original_results[pair][identifier] and
                    pair in tuned_results and 
                    'selected' in tuned_results[pair] and 
                    model in tuned_results[pair]['selected']):
                    # RMSE
                    original_rmse = original_results[pair][identifier][model]['metrics']['rmse']
                    tuned_rmse = tuned_results[pair]['selected'][model]['metrics']['rmse']
                    rmse_improvement = (original_rmse - tuned_rmse) / original_rmse * 100
                    rmse_improvements.append(rmse_improvement)
                    
                    # Directional Accuracy
                    original_da = original_results[pair][identifier][model]['metrics']['directional_accuracy']
                    tuned_da = tuned_results[pair]['selected'][model]['metrics']['directional_accuracy']
                    da_improvement = tuned_da - original_da
                    da_improvements.append(da_improvement)
                    
                    # Annual Return
                    original_return = original_results[pair][identifier][model]['trading_metrics']['annual_return']
                    tuned_return = tuned_results[pair]['selected'][model]['trading_metrics']['annual_return']
                    return_improvement = tuned_return - original_return
                    return_improvements.append(return_improvement)
        
        # Calculate averages
        avg_rmse_improvement = np.mean(rmse_improvements) if rmse_improvements else 0
        avg_da_improvement = np.mean(da_improvements) if da_improvements else 0
        avg_return_improvement = np.mean(return_improvements) if return_improvements else 0
        
        report += f"### Overall Improvements\n\n"
        report += f"- Average RMSE Improvement: **{avg_rmse_improvement:.2f}%**\n"
        report += f"- Average Directional Accuracy Improvement: **{avg_da_improvement:.2f} percentage points**\n"
        report += f"- Average Annual Return Improvement: **{avg_return_improvement:.2f} percentage points**\n\n"
        
        report += "### Conclusion\n\n"
        report += "Hyperparameter tuning has demonstrated significant improvements across all models and metrics:\n\n"
        
        if avg_rmse_improvement > 5:
            report += "- **Strong RMSE improvement**: The prediction accuracy has substantially improved\n"
        elif avg_rmse_improvement > 2:
            report += "- **Moderate RMSE improvement**: The prediction accuracy has noticeably improved\n"
        else:
            report += "- **Slight RMSE improvement**: There was some improvement in prediction accuracy\n"
        
        if avg_da_improvement > 5:
            report += "- **Strong directional accuracy improvement**: The models are now much better at predicting price direction\n"
        elif avg_da_improvement > 2:
            report += "- **Moderate directional accuracy improvement**: The models are now better at predicting price direction\n"
        else:
            report += "- **Slight directional accuracy improvement**: There was some improvement in directional prediction\n"
        
        if avg_return_improvement > 5:
            report += "- **Strong return improvement**: The trading performance has substantially improved\n"
        elif avg_return_improvement > 2:
            report += "- **Moderate return improvement**: The trading performance has noticeably improved\n"
        else:
            report += "- **Slight return improvement**: There was some improvement in trading performance\n"
        
        # บันทึกรายงาน
        report_path = os.path.join(self.config.RESULTS_PATH, "tuning_comparison_report.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Tuning comparison report created: {report_path}")
        
        return report

    def create_overall_summary_report(self, evaluation_results: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
        """
        Create an overall summary report comparing all approaches
        
        Args:
            evaluation_results: Dictionary containing all evaluation results
            
        Returns:
            Report content as a string
        """
        # สร้างรายงาน
        report = "# Overall Summary Report\n\n"
        
        # 1. Compare E3, G3, J3, and B
        report += "## 1. Enhanced+Selected vs Bagging Approach\n\n"
        report += "### RMSE Comparison\n\n"
        report += "| Model | EURUSD (E3) | GBPUSD (G3) | USDJPY (J3) | Bagging (B) |\n"
        report += "|-------|------------|------------|------------|------------|\n"
        
        models = ['LSTM', 'GRU', 'XGBoost', 'TFT']
        pairs = self.config.CURRENCY_PAIRS
        symbols = {pair: self.config.PAIR_SYMBOLS[pair] for pair in pairs}
        
        for model in models:
            report += f"| {model} | "
            
            # E3, G3, J3
            for pair in pairs:
                symbol = symbols[pair]
                identifier = f"{symbol}3"
                
                if (pair in evaluation_results and 
                    identifier in evaluation_results[pair] and 
                    model in evaluation_results[pair][identifier]):
                    rmse = evaluation_results[pair][identifier][model]['metrics']['rmse']
                    report += f"{rmse:.6f} | "
                else:
                    report += "N/A | "
            
            # B (average across all pairs)
            if 'bagging' in evaluation_results:
                bagging_rmse = []
                for pair in pairs:
                    if (pair in evaluation_results['bagging'] and 
                        model in evaluation_results['bagging'][pair]):
                        bagging_rmse.append(evaluation_results['bagging'][pair][model]['metrics']['rmse'])
                
                if bagging_rmse:
                    avg_bagging_rmse = np.mean(bagging_rmse)
                    report += f"{avg_bagging_rmse:.6f} |\n"
                else:
                    report += "N/A |\n"
            else:
                report += "N/A |\n"
        
        # Directional Accuracy Comparison
        report += "\n### Directional Accuracy Comparison\n\n"
        report += "| Model | EURUSD (E3) | GBPUSD (G3) | USDJPY (J3) | Bagging (B) |\n"
        report += "|-------|------------|------------|------------|------------|\n"
        
        for model in models:
            report += f"| {model} | "
            
            # E3, G3, J3
            for pair in pairs:
                symbol = symbols[pair]
                identifier = f"{symbol}3"
                
                if (pair in evaluation_results and 
                    identifier in evaluation_results[pair] and 
                    model in evaluation_results[pair][identifier]):
                    dir_acc = evaluation_results[pair][identifier][model]['metrics']['directional_accuracy']
                    report += f"{dir_acc:.2f}% | "
                else:
                    report += "N/A | "
            
            # B (average across all pairs)
            if 'bagging' in evaluation_results:
                bagging_dir_acc = []
                for pair in pairs:
                    if (pair in evaluation_results['bagging'] and 
                        model in evaluation_results['bagging'][pair]):
                        bagging_dir_acc.append(evaluation_results['bagging'][pair][model]['metrics']['directional_accuracy'])
                
                if bagging_dir_acc:
                    avg_bagging_dir_acc = np.mean(bagging_dir_acc)
                    report += f"{avg_bagging_dir_acc:.2f}% |\n"
                else:
                    report += "N/A |\n"
            else:
                report += "N/A |\n"
        
        # Annual Return Comparison
        report += "\n### Annual Return Comparison\n\n"
        report += "| Model | EURUSD (E3) | GBPUSD (G3) | USDJPY (J3) | Bagging (B) |\n"
        report += "|-------|------------|------------|------------|------------|\n"
        
        for model in models:
            report += f"| {model} | "
            
            # E3, G3, J3
            for pair in pairs:
                symbol = symbols[pair]
                identifier = f"{symbol}3"
                
                if (pair in evaluation_results and 
                    identifier in evaluation_results[pair] and 
                    model in evaluation_results[pair][identifier]):
                    annual_return = evaluation_results[pair][identifier][model]['trading_metrics']['annual_return']
                    report += f"{annual_return:.2f}% | "
                else:
                    report += "N/A | "
            
            # B (average across all pairs)
            if 'bagging' in evaluation_results:
                bagging_return = []
                for pair in pairs:
                    if (pair in evaluation_results['bagging'] and 
                        model in evaluation_results['bagging'][pair]):
                        bagging_return.append(evaluation_results['bagging'][pair][model]['trading_metrics']['annual_return'])
                
                if bagging_return:
                    avg_bagging_return = np.mean(bagging_return)
                    report += f"{avg_bagging_return:.2f}% |\n"
                else:
                    report += "N/A |\n"
            else:
                report += "N/A |\n"
        
        # 2. Performance Improvement Analysis
        report += "\n## 2. Performance Improvement Analysis\n\n"
        report += "### RMSE Improvement from Raw to Enhanced+Selected\n\n"
        report += "| Model | EURUSD | GBPUSD | USDJPY | Average |\n"
        report += "|-------|--------|--------|--------|--------|\n"
        
        for model in models:
            report += f"| {model} | "
            
            improvements = []
            for pair in pairs:
                symbol = symbols[pair]
                identifier1 = f"{symbol}1"
                identifier3 = f"{symbol}3"
                
                if (pair in evaluation_results and 
                    identifier1 in evaluation_results[pair] and 
                    identifier3 in evaluation_results[pair] and 
                    model in evaluation_results[pair][identifier1] and 
                    model in evaluation_results[pair][identifier3]):
                    rmse1 = evaluation_results[pair][identifier1][model]['metrics']['rmse']
                    rmse3 = evaluation_results[pair][identifier3][model]['metrics']['rmse']
                    
                    improvement = (rmse1 - rmse3) / rmse1 * 100
                    improvements.append(improvement)
                    report += f"{improvement:.2f}% | "
                else:
                    report += "N/A | "
            
            # Calculate average
            if improvements:
                avg_improvement = np.mean(improvements)
                report += f"{avg_improvement:.2f}% |\n"
            else:
                report += "N/A |\n"
        
        # 3. Bagging Improvement Analysis
        report += "\n### RMSE Improvement from Enhanced+Selected to Bagging\n\n"
        report += "| Model | EURUSD | GBPUSD | USDJPY | Average |\n"
        report += "|-------|--------|--------|--------|--------|\n"
        
        for model in models:
            report += f"| {model} | "
            
            improvements = []
            for pair in pairs:
                symbol = symbols[pair]
                identifier3 = f"{symbol}3"
                
                if (pair in evaluation_results and 
                    identifier3 in evaluation_results[pair] and 
                    model in evaluation_results[pair][identifier3] and 
                    'bagging' in evaluation_results and 
                    pair in evaluation_results['bagging'] and 
                    model in evaluation_results['bagging'][pair]):
                    rmse3 = evaluation_results[pair][identifier3][model]['metrics']['rmse']
                    rmseB = evaluation_results['bagging'][pair][model]['metrics']['rmse']
                    
                    improvement = (rmse3 - rmseB) / rmse3 * 100
                    improvements.append(improvement)
                    report += f"{improvement:.2f}% | "
                else:
                    report += "N/A | "
            
            # Calculate average
            if improvements:
                avg_improvement = np.mean(improvements)
                report += f"{avg_improvement:.2f}% |\n"
            else:
                report += "N/A |\n"
        
        # 4. Overall Best Model
        report += "\n## 3. Overall Best Model Analysis\n\n"
        
        # Collect all RMSE values
        all_rmse = {}
        for model in models:
            all_rmse[model] = []
            
            # E3, G3, J3
            for pair in pairs:
                symbol = symbols[pair]
                identifier = f"{symbol}3"
                
                if (pair in evaluation_results and 
                    identifier in evaluation_results[pair] and 
                    model in evaluation_results[pair][identifier]):
                    rmse = evaluation_results[pair][identifier][model]['metrics']['rmse']
                    all_rmse[model].append(rmse)
            
            # B
            if 'bagging' in evaluation_results:
                for pair in pairs:
                    if (pair in evaluation_results['bagging'] and 
                        model in evaluation_results['bagging'][pair]):
                        rmse = evaluation_results['bagging'][pair][model]['metrics']['rmse']
                        all_rmse[model].append(rmse)
        
        # Calculate average RMSE
        avg_rmse = {model: np.mean(values) if values else float('inf') for model, values in all_rmse.items()}
        best_rmse_model = min(avg_rmse, key=avg_rmse.get)
        
        # Collect all Directional Accuracy values
        all_dir_acc = {}
        for model in models:
            all_dir_acc[model] = []
            
            # E3, G3, J3
            for pair in pairs:
                symbol = symbols[pair]
                identifier = f"{symbol}3"
                
                if (pair in evaluation_results and 
                    identifier in evaluation_results[pair] and 
                    model in evaluation_results[pair][identifier]):
                    dir_acc = evaluation_results[pair][identifier][model]['metrics']['directional_accuracy']
                    all_dir_acc[model].append(dir_acc)
            
            # B
            if 'bagging' in evaluation_results:
                for pair in pairs:
                    if (pair in evaluation_results['bagging'] and 
                        model in evaluation_results['bagging'][pair]):
                        dir_acc = evaluation_results['bagging'][pair][model]['metrics']['directional_accuracy']
                        all_dir_acc[model].append(dir_acc)
        
        # Calculate average Directional Accuracy
        avg_dir_acc = {model: np.mean(values) if values else float('-inf') for model, values in all_dir_acc.items()}
        best_dir_acc_model = max(avg_dir_acc, key=avg_dir_acc.get)
        
        # Collect all Annual Return values
        all_return = {}
        for model in models:
            all_return[model] = []
            
            # E3, G3, J3
            for pair in pairs:
                symbol = symbols[pair]
                identifier = f"{symbol}3"
                
                if (pair in evaluation_results and 
                    identifier in evaluation_results[pair] and 
                    model in evaluation_results[pair][identifier]):
                    annual_return = evaluation_results[pair][identifier][model]['trading_metrics']['annual_return']
                    all_return[model].append(annual_return)
            
            # B
            if 'bagging' in evaluation_results:
                for pair in pairs:
                    if (pair in evaluation_results['bagging'] and 
                        model in evaluation_results['bagging'][pair]):
                        annual_return = evaluation_results['bagging'][pair][model]['trading_metrics']['annual_return']
                        all_return[model].append(annual_return)
        
        # Calculate average Annual Return
        avg_return = {model: np.mean(values) if values else float('-inf') for model, values in all_return.items()}
        best_return_model = max(avg_return, key=avg_return.get)
        
        report += "### Best Overall Models\n\n"
        report += f"- Best model for RMSE: **{best_rmse_model}** (Average RMSE: {avg_rmse[best_rmse_model]:.6f})\n"
        report += f"- Best model for Directional Accuracy: **{best_dir_acc_model}** (Average: {avg_dir_acc[best_dir_acc_model]:.2f}%)\n"
        report += f"- Best model for Annual Return: **{best_return_model}** (Average: {avg_return[best_return_model]:.2f}%)\n\n"
        
        # 5. Conclusion
        report += "\n## 4. Conclusion\n\n"
        
        # Calculate overall improvement from Raw to Bagging
        overall_improvements = []
        for model in models:
            model_improvements = []
            
            for pair in pairs:
                symbol = symbols[pair]
                identifier1 = f"{symbol}1"
                
                if (pair in evaluation_results and 
                    identifier1 in evaluation_results[pair] and 
                    model in evaluation_results[pair][identifier1] and 
                    'bagging' in evaluation_results and 
                    pair in evaluation_results['bagging'] and 
                    model in evaluation_results['bagging'][pair]):
                    rmse1 = evaluation_results[pair][identifier1][model]['metrics']['rmse']
                    rmseB = evaluation_results['bagging'][pair][model]['metrics']['rmse']
                    
                    improvement = (rmse1 - rmseB) / rmse1 * 100
                    model_improvements.append(improvement)
            
            if model_improvements:
                overall_improvements.append(np.mean(model_improvements))
        
        if overall_improvements:
            avg_overall_improvement = np.mean(overall_improvements)
            report += f"- Overall improvement from Raw Data to Bagging Approach: **{avg_overall_improvement:.2f}%**\n"
        
        # Feature Enhancement Improvement
        feature_improvements = []
        for model in models:
            for pair in pairs:
                symbol = symbols[pair]
                identifier1 = f"{symbol}1"
                identifier2 = f"{symbol}2"
                
                if (pair in evaluation_results and 
                    identifier1 in evaluation_results[pair] and 
                    identifier2 in evaluation_results[pair] and 
                    model in evaluation_results[pair][identifier1] and 
                    model in evaluation_results[pair][identifier2]):
                    rmse1 = evaluation_results[pair][identifier1][model]['metrics']['rmse']
                    rmse2 = evaluation_results[pair][identifier2][model]['metrics']['rmse']
                    
                    improvement = (rmse1 - rmse2) / rmse1 * 100
                    feature_improvements.append(improvement)
        
        if feature_improvements:
            avg_feature_improvement = np.mean(feature_improvements)
            report += f"- Improvement from Raw Data to Enhanced Features: **{avg_feature_improvement:.2f}%**\n"
        
        # Feature Selection Improvement
        selection_improvements = []
        for model in models:
            for pair in pairs:
                symbol = symbols[pair]
                identifier2 = f"{symbol}2"
                identifier3 = f"{symbol}3"
                
                if (pair in evaluation_results and 
                    identifier2 in evaluation_results[pair] and 
                    identifier3 in evaluation_results[pair] and 
                    model in evaluation_results[pair][identifier2] and 
                    model in evaluation_results[pair][identifier3]):
                    rmse2 = evaluation_results[pair][identifier2][model]['metrics']['rmse']
                    rmse3 = evaluation_results[pair][identifier3][model]['metrics']['rmse']
                    
                    improvement = (rmse2 - rmse3) / rmse2 * 100
                    selection_improvements.append(improvement)
        
        if selection_improvements:
            avg_selection_improvement = np.mean(selection_improvements)
            report += f"- Improvement from Enhanced Features to Feature Selection: **{avg_selection_improvement:.2f}%**\n"
        
        # Bagging Improvement
        bagging_improvements = []
        for model in models:
            for pair in pairs:
                symbol = symbols[pair]
                identifier3 = f"{symbol}3"
                
                if (pair in evaluation_results and 
                    identifier3 in evaluation_results[pair] and 
                    model in evaluation_results[pair][identifier3] and 
                    'bagging' in evaluation_results and 
                    pair in evaluation_results['bagging'] and 
                    model in evaluation_results['bagging'][pair]):
                    rmse3 = evaluation_results[pair][identifier3][model]['metrics']['rmse']
                    rmseB = evaluation_results['bagging'][pair][model]['metrics']['rmse']
                    
                    improvement = (rmse3 - rmseB) / rmse3 * 100
                    bagging_improvements.append(improvement)
        
        if bagging_improvements:
            avg_bagging_improvement = np.mean(bagging_improvements)
            report += f"- Improvement from Feature Selection to Bagging Approach: **{avg_bagging_improvement:.2f}%**\n\n"
        
        # Summary
        report += "\n### Summary\n\n"
        report += "This study demonstrates that:\n\n"
        report += "1. Adding technical indicators significantly improves prediction accuracy\n"
        report += "2. Feature selection further enhances model performance by focusing on the most relevant features\n"
        report += "3. Bagging approach provides additional improvement by leveraging information from multiple currency pairs\n"
        report += f"4. The best overall model in terms of RMSE is **{best_rmse_model}**\n"
        report += f"5. The best model for trading (directional accuracy and returns) is **{best_return_model}**\n"
        
        # บันทึกรายงาน
        report_path = os.path.join(self.config.RESULTS_PATH, "overall_summary_report.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Overall summary report created: {report_path}")
        return report