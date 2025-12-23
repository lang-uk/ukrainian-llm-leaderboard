import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from typing import Dict, List, Any, Tuple, Optional
from huggingface_hub import snapshot_download
from pathlib import Path

# Define the key benchmarks to track from the JSON results file
# Define the key benchmarks to track from the JSON results file
MAIN_BENCHMARKS = {
    "belebele_ukr_Cyrl": {
        "metric": "acc",
        "name": "Belebele Ukrainian",
        "scale": [0, 1],
    },
    "global_mmlu_full_uk": {"metric": "acc", "name": "MMLU Ukrainian", "scale": [0, 1]},
    "flores_uk": {"metric": "bleu", "name": "FLORES Ukrainian", "scale": [0, 40]},
    "long_flores_uk": {
        "metric": "bleu",
        "name": "Long FLORES Ukrainian",
        "scale": [0, 40],
    },
    "squad_uk": {"metric": "f1", "name": "SQuAD Ukrainian", "scale": [0, 100]},
    "xlsum_uk": {"metric": "bleu", "name": "XLSum Ukrainian", "scale": [0, 30]},
    "triviaqa_uk": {
        "metric": "exact_match",
        "name": "TriviaQA Ukrainian",
        "scale": [0, 1],
    },
    "arc_challenge_uk": {
        "metric": "exact_match",
        "name": "ARC Challenge Ukrainian",
        "scale": [0, 1],
    },
    "arc_easy_uk": {"metric": "acc", "name": "ARC Easy Ukrainian", "scale": [0, 1]},
    "winogrande_uk": {"metric": "acc", "name": "Winogrande Ukrainian", "scale": [0, 1]},
    "gsm8k_uk": {"metric": "exact_match", "name": "GSM8K Ukrainian", "scale": [0, 1]},
    "ifeval_uk": {
        "metric": "prompt_level_strict_acc",
        "name": "IFEval Ukrainian",
        "scale": [0, 1],
    },
    "wmt_en_uk": {"metric": "bleu", "name": "WMT EN→UK", "scale": [0, 40]},
    "zno_uk_geography": {"metric": "exact", "name": "ZNO Geography", "scale": [0, 1]},
    "zno_uk_history": {"metric": "exact", "name": "ZNO History", "scale": [0, 1]},
    "zno_uk_language_and_literature": {
        "metric": "exact",
        "name": "ZNO Language & Literature",
        "scale": [0, 1],
    },
    "zno_uk_math": {"metric": "exact", "name": "ZNO Math", "scale": [0, 1]},
}

# MMLU - Only use aggregate score, no subcategories
MMLU_BENCHMARKS = {}

# FLORES Language Pair benchmarks - ONLY English-Ukrainian pairs
FLORES_BENCHMARKS = {
    "flores_en-uk": {"metric": "bleu", "name": "FLORES EN→UK", "scale": [0, 40]},
    "flores_uk-en": {"metric": "bleu", "name": "FLORES UK→EN", "scale": [0, 40]},
}

# Long FLORES Language Pair benchmarks - ONLY English-Ukrainian pairs
LONG_FLORES_BENCHMARKS = {
    "long_flores_en-uk": {
        "metric": "bleu",
        "name": "Long FLORES EN→UK",
        "scale": [0, 40],
    },
    "long_flores_uk-en": {
        "metric": "bleu",
        "name": "Long FLORES UK→EN",
        "scale": [0, 40],
    },
}

# Combine all benchmarks for detailed view
ALL_BENCHMARKS = {
    **MAIN_BENCHMARKS,
    **MMLU_BENCHMARKS,
    **FLORES_BENCHMARKS,
    **LONG_FLORES_BENCHMARKS,
}


def extract_model_name(file_path: str) -> str:
    """Extract model name from the file path."""
    # Format: eval-results/<model_name>/results_*.json
    parts = file_path.split(os.sep)
    if len(parts) >= 2:
        return parts[-2]  # The second-to-last element should be the model name
    return os.path.basename(file_path).replace("results_", "").replace(".json", "")


def download_benchmark_dataset(
    repo_id: str = "lang-uk/ukrainian-llm-leaderboard-results",
    local_dir: str = "./eval-results",
    token: Optional[str] = None,
    force_download: bool = False,
) -> str:
    """
    Download benchmark results dataset from Hugging Face Hub if local directory doesn't exist or is empty.

    Args:
        repo_id: The Hugging Face repository ID containing the benchmark results
        local_dir: Local directory to download the files to
        token: Hugging Face token (optional, for private repos)
        force_download: If True, download even if directory exists and has files

    Returns:
        Path to the directory (either existing or downloaded)
    """
    local_path = Path(local_dir)

    # Check if directory exists and has files
    if not force_download and local_path.exists() and any(local_path.iterdir()):
        print(
            f"Directory {local_dir} already exists and contains files. Using existing data."
        )
        return str(local_path)

    try:
        # Create local directory if it doesn't exist
        local_path.mkdir(parents=True, exist_ok=True)

        print(f"Downloading benchmark dataset from {repo_id} to {local_dir}...")

        # Download the entire repository
        downloaded_path = snapshot_download(
            repo_id=repo_id, local_dir=local_dir, token=token, repo_type="dataset"
        )

        print(f"Dataset downloaded successfully to: {downloaded_path}")
        return downloaded_path

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print(f"Using local directory: {local_dir}")
        return local_dir


def load_results(results_dir: str = "eval-results") -> List[Dict[str, Any]]:
    """Load all results from JSON files in the results directory."""
    all_results = []

    # Check if directory exists and has content, download if not
    if not os.path.exists(results_dir) or not os.listdir(results_dir):
        print(
            f"Results directory {results_dir} not found or empty. Attempting to download..."
        )
        results_dir = download_benchmark_dataset(local_dir=results_dir)

    results_dir = os.path.join(results_dir, "aggregated")

    # Find all results files
    pattern = os.path.join(results_dir, "**", "results*.json")
    result_files = glob.glob(pattern, recursive=True)

    if not result_files:
        print(f"No result files found in {results_dir}")
        # Create a sample result for demonstration if no files found
        return [{"model_name": "No models found - Add files to eval-results directory"}]

    for file_path in result_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # model_name = data.get("model_name", extract_model_name(file_path))
            parent_folder = os.path.basename(
                os.path.dirname(os.path.dirname(file_path))
            )
            model_name = data.get(
                "model_name",
                extract_model_name(file_path),
            )

            # Extract relevant metrics
            model_results = {"model_name": model_name}

            model_in_results = any([model_name == k["model_name"] for k in all_results])
            if model_in_results:
                print(f"Duplicate model name found: {model_name}. Skipping.")
                continue

            # Process all benchmark categories
            for benchmark_dict in [ALL_BENCHMARKS]:
                for benchmark, config in benchmark_dict.items():
                    metric = config["metric"]
                    # Check if benchmark exists in results
                    if benchmark in data.get("results", {}):
                        result = data["results"][benchmark]
                        # Handle different metric formats in the JSON
                        if f"{metric},none" in result:
                            model_results[benchmark] = result[f"{metric},none"]
                        elif f"{metric},remove_whitespace" in result:
                            model_results[benchmark] = result[
                                f"{metric},remove_whitespace"
                            ]
                        elif f"{metric},flexible-extract" in result:
                            model_results[benchmark] = result[
                                f"{metric},flexible-extract"
                            ]
                        elif f"{metric},strict-match" in result:
                            model_results[benchmark] = result[f"{metric},strict-match"]
                        elif metric in result:
                            model_results[benchmark] = result[metric]

            # Calculate mean of EN↔UK pairs for flores_uk
            if "flores_en-uk" in model_results and "flores_uk-en" in model_results:
                model_results["flores_uk"] = (
                    model_results["flores_en-uk"] + model_results["flores_uk-en"]
                ) / 2

            # Calculate mean of EN↔UK pairs for long_flores_uk
            if (
                "long_flores_en-uk" in model_results
                and "long_flores_uk-en" in model_results
            ):
                model_results["long_flores_uk"] = (
                    model_results["long_flores_en-uk"]
                    + model_results["long_flores_uk-en"]
                ) / 2

            all_results.append(model_results)

        except Exception as e:
            raise e
            print(f"Error loading {file_path}: {e}")

    return all_results


def create_dataframe(
    results: List[Dict[str, Any]],
    benchmark_set: Dict[str, Dict],
    normalize_scores: bool = True,
) -> pd.DataFrame:
    """Create a DataFrame from the results with the specified benchmarks."""
    if not results:
        return pd.DataFrame(columns=["Model"])

    df = pd.DataFrame(results)

    # Prepare list of columns to include
    columns = ["model_name"]
    column_mapping = {"model_name": "Model"}

    # Add benchmarks that exist in the results
    for benchmark, config in benchmark_set.items():
        if benchmark in df.columns:
            columns.append(benchmark)
            column_mapping[benchmark] = config["name"]

    # Filter columns that exist
    existing_columns = [col for col in columns if col in df.columns]

    # Create the dataframe with selected columns
    if len(existing_columns) <= 1:  # Only model_name column exists
        return pd.DataFrame({"Model": df["model_name"]})

    result_df = df[existing_columns].rename(columns=column_mapping)

    # Normalize scores for better readability if requested
    if normalize_scores:
        for col in result_df.columns:
            if col != "Model":
                # Check if values are in [0,1] range (except for BLEU which might be higher)
                if (
                    col.startswith("MMLU")
                    or col == "Belebele Ukrainian"
                    or col == "SQuAD Ukrainian"
                    or col == "ARC Easy Ukrainian"
                    or col == "Winogrande Ukrainian"
                    or col == "IFEval Ukrainian"
                ) and result_df[col].mean() < 1:
                    result_df[col] = (result_df[col] * 100).round(2)
                else:
                    result_df[col] = result_df[col].round(2)

    return result_df


def calculate_average_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the average rank of each model across all metrics."""
    if len(df.columns) <= 1:  # Only Model column
        return df

    # Get metric columns (all except Model)
    metric_columns = [col for col in df.columns if col != "Model"]

    # Calculate ranks for each metric (higher is better)
    for col in metric_columns:
        rank_col = f"{col}_rank"
        df[rank_col] = df[col].rank(ascending=False, method="min")

    # Calculate average rank
    rank_columns = [f"{col}_rank" for col in metric_columns]
    df["Average Rank"] = df[rank_columns].mean(axis=1).round(2)

    # Drop individual rank columns
    df = df.drop(columns=rank_columns)

    # Sort by average rank (lower is better)
    df = df.sort_values("Average Rank", ascending=True)

    return df


def create_relative_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate relative performance as percentage of the best model's score."""
    result_df = df.copy()

    for col in result_df.columns:
        if col not in ["Model", "Average Rank"]:
            max_score = result_df[col].max()
            if max_score > 0:
                result_df[col] = ((result_df[col] / max_score) * 100).round(2)

    return result_df


def create_radar_chart(df: pd.DataFrame, selected_models: List[str]) -> plt.Figure:
    """Create a radar chart comparing multiple models."""
    if not selected_models or len(selected_models) == 0:
        return None

    # Get metric columns (excluding Model and Average Rank)
    metric_columns = [col for col in df.columns if col not in ["Model", "Average Rank"]]

    if len(metric_columns) == 0:
        return None

    # Filter data for selected models
    plot_df = df[df["Model"].isin(selected_models)]

    # Number of variables
    num_vars = len(metric_columns)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    # Get scales for normalization
    scales = {}
    for col in metric_columns:
        # Find the corresponding benchmark configuration
        benchmark_key = None
        for bench_dict in [MAIN_BENCHMARKS, FLORES_BENCHMARKS, LONG_FLORES_BENCHMARKS]:
            for key, config in bench_dict.items():
                if config["name"] == col:
                    benchmark_key = key
                    break
            if benchmark_key:
                break

        if benchmark_key and benchmark_key in ALL_BENCHMARKS:
            scales[col] = ALL_BENCHMARKS[benchmark_key]["scale"]
        else:
            # Fallback to data-based scale
            scales[col] = [0, df[col].max() * 1.1]

    # Normalize values to 0-100 scale for radar chart
    normalized_df = plot_df.copy()
    for col in metric_columns:
        min_val, max_val = scales[col]
        normalized_df[col] = (
            (plot_df[col] - min_val) / (max_val - min_val) * 100
        ).clip(0, 100)

    # Plot each model
    for idx, model in enumerate(selected_models):
        model_data = normalized_df[normalized_df["Model"] == model]
        if len(model_data) == 0:
            continue

        values = model_data[metric_columns].values.flatten().tolist()
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, "o-", linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.15)

    # Set scale to 0-100
    ax.set_ylim(0, 100)

    # Fix axis to go in the right order
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_columns, size=8)

    # Add concentric circles for reference
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], size=6)
    ax.grid(True)

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.title("Model Comparison (Normalized to Scale)", size=14, y=1.08)
    plt.tight_layout()

    return fig


def create_bar_chart(df: pd.DataFrame, metric: str) -> plt.Figure:
    """Create a bar chart for a specific metric."""
    if metric not in df.columns or metric == "Model":
        return None

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.3)))

    # Sort by the metric
    plot_df = df.sort_values(metric, ascending=True)

    # Create horizontal bar chart
    bars = ax.barh(plot_df["Model"], plot_df[metric])

    # Color bars with a gradient
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(plot_df)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Set x-axis scale based on benchmark configuration
    benchmark_key = None
    for bench_dict in [MAIN_BENCHMARKS, FLORES_BENCHMARKS, LONG_FLORES_BENCHMARKS]:
        for key, config in bench_dict.items():
            if config["name"] == metric:
                benchmark_key = key
                break
        if benchmark_key:
            break

    if False: # TODO: proper fix for scales # benchmark_key and benchmark_key in ALL_BENCHMARKS:
        min_val, max_val = ALL_BENCHMARKS[benchmark_key]["scale"]
        ax.set_xlim(min_val, max_val)

    ax.set_xlabel(metric)
    ax.set_title(f"Model Performance on {metric}")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return fig


def create_leaderboard_app() -> gr.Blocks:
    """Create the Gradio interface for the leaderboard."""
    # Load results

    results = load_results("eval-results")

    # Create main dataframe
    main_df = create_dataframe(results, MAIN_BENCHMARKS)
    main_df = calculate_average_rank(main_df)

    # Create detailed dataframe
    detailed_df = create_dataframe(results, ALL_BENCHMARKS)
    detailed_df = calculate_average_rank(detailed_df)

    # Create MMLU dataframe (only aggregate, no subcategories)
    mmlu_df = create_dataframe(
        results,
        {"global_mmlu_full_uk": MAIN_BENCHMARKS["global_mmlu_full_uk"]},
    )
    mmlu_df = calculate_average_rank(mmlu_df)

    # Create FLORES dataframe (only en-uk and uk-en)
    flores_df = create_dataframe(
        results,
        {**{"flores_uk": MAIN_BENCHMARKS["flores_uk"]}, **FLORES_BENCHMARKS},
    )

    flores_df = calculate_average_rank(flores_df)

    # Create Long FLORES dataframe (only en-uk and uk-en)
    long_flores_df = create_dataframe(
        results,
        {
            **{"long_flores_uk": MAIN_BENCHMARKS["long_flores_uk"]},
            **LONG_FLORES_BENCHMARKS,
        },
    )
    long_flores_df = calculate_average_rank(long_flores_df)

    # Create Gradio interface
    with gr.Blocks(
        title="Ukrainian Language Model Leaderboard",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
        # 🇺🇦 Ukrainian Language Model Leaderboard
        
        Welcome to the Ukrainian Language Model Leaderboard! This dashboard displays performance metrics 
        for various language models on Ukrainian language benchmarks.
        """
        )

        # Main Leaderboard Tab
        with gr.Tab("📊 Main Leaderboard"):
            gr.Markdown(
                """
            ## Main Performance Metrics
            
            This table shows the core benchmarks for evaluating Ukrainian language understanding and generation.
            Models are ranked by their average performance across all metrics.
            
            **Note:** For FLORES benchmarks, only English↔Ukrainian pairs are shown. For MMLU, only the aggregate score is displayed.
            """
            )

            with gr.Row():
                with gr.Column(scale=4):
                    main_table = gr.Dataframe(
                        value=main_df.sort_values("Average Rank", ascending=True),
                        label="Leaderboard",
                        interactive=False,
                        wrap=False,
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### Display Options")
                    main_sort_by = gr.Dropdown(
                        choices=main_df.columns.tolist(),
                        value="Average Rank",
                        label="Sort by",
                    )
                    main_sort_asc = gr.Checkbox(value=True, label="Ascending Order")
                    main_show_relative = gr.Checkbox(
                        value=False, label="Show Relative Scores (%)"
                    )
                    refresh_btn = gr.Button("🔄 Refresh Data")

        # Detailed Benchmarks Tab
        with gr.Tab("📈 Detailed Benchmarks"):
            gr.Markdown(
                """
            ## Detailed Performance Breakdown
            
            Explore performance on specific benchmark categories.
            
            **Note:** MMLU shows only the aggregate score. FLORES and Long FLORES show only English↔Ukrainian translation pairs.
            """
            )

            with gr.Row():
                benchmark_category = gr.Radio(
                    choices=[
                        "All Benchmarks",
                        "MMLU (Aggregate Only)",
                        "FLORES Translation Pairs (EN↔UK)",
                        "Long FLORES Translation Pairs (EN↔UK)",
                    ],
                    value="All Benchmarks",
                    label="Select Category",
                )
                remove_dupes = gr.Checkbox(value=False, label="Remove Duplicate Models")

            detailed_table = gr.Dataframe(
                value=detailed_df.sort_values("Average Rank", ascending=True),
                label="Detailed Results",
                interactive=False,
                wrap=False,
            )

            with gr.Row():
                detailed_sort_by = gr.Dropdown(
                    choices=detailed_df.columns.tolist(),
                    value="Average Rank",
                    label="Sort by",
                )
                detailed_sort_asc = gr.Checkbox(value=True, label="Ascending Order")

        # Model Comparison Tab
        with gr.Tab("🔍 Model Comparison"):
            gr.Markdown(
                """
            ## Compare Models
            
            Select multiple models to compare their performance across all benchmarks using a radar chart.
            """
            )

            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=(
                        main_df["Model"].tolist() if "Model" in main_df.columns else []
                    ),
                    multiselect=True,
                    label="Select Models to Compare",
                )
                compare_btn = gr.Button("Generate Comparison")

            with gr.Row():
                comparison_chart = gr.Plot(label="Radar Chart Comparison")

            comparison_table = gr.Dataframe(
                label="Comparison Table",
                interactive=False,
                wrap=False,
            )

        # Visualizations Tab
        with gr.Tab("📊 Visualizations"):
            with gr.Row():
                with gr.Column():
                    metric_selector = gr.Dropdown(
                        choices=[col for col in main_df.columns if col != "Model"],
                        value=main_df.columns[1] if len(main_df.columns) > 1 else None,
                        label="Select Metric to Visualize",
                    )
                    viz_btn = gr.Button("Generate Visualization")

            with gr.Row():
                metric_plot = gr.Plot(label="Benchmark Performance")

            gr.Markdown(
                """
            ## Visualizations
            
            Select a metric to generate a bar chart showing the performance of all models on that specific benchmark.
            """
            )

        # Update functions
        def update_main_table(sort_by, sort_asc, show_relative):
            df = main_df.copy()

            if show_relative:
                df = create_relative_scores(df)

            if sort_by in df.columns:
                df = df.sort_values(sort_by, ascending=sort_asc)

            return df

        def update_detailed_table(category, sort_by, sort_asc, remove_duplicates):
            if category == "All Benchmarks":
                df = detailed_df.copy()
            elif category == "MMLU (Aggregate Only)":
                df = mmlu_df.copy()
            elif category == "FLORES Translation Pairs (EN↔UK)":
                df = flores_df.copy()
            elif category == "Long FLORES Translation Pairs (EN↔UK)":
                df = long_flores_df.copy()
            else:
                df = detailed_df.copy()

            if remove_duplicates and "Model" in df.columns:
                # Keep only the first occurrence of each model
                df = df.drop_duplicates(subset=["Model"])

            if sort_by in df.columns:
                df = df.sort_values(sort_by, ascending=sort_asc)

            return df

        def update_comparison(models):
            if not models or len(models) == 0:
                return None, pd.DataFrame()

            # Create comparison chart
            chart = create_radar_chart(main_df, models)

            # Create comparison table
            table = main_df[main_df["Model"].isin(models)].copy()

            return chart, table

        def update_visualization(metric):
            if not metric:
                return None

            chart = create_bar_chart(main_df, metric)
            return chart

        def refresh_data():
            nonlocal results, main_df, detailed_df, mmlu_df, flores_df, long_flores_df
            results = load_results()

            # Recreate all dataframes
            main_df = create_dataframe(results, MAIN_BENCHMARKS)
            main_df = calculate_average_rank(main_df)

            detailed_df = create_dataframe(results, ALL_BENCHMARKS)
            detailed_df = calculate_average_rank(detailed_df)

            mmlu_df = create_dataframe(
                results,
                {"global_mmlu_full_uk": MAIN_BENCHMARKS["global_mmlu_full_uk"]},
            )
            mmlu_df = calculate_average_rank(mmlu_df)

            flores_df = create_dataframe(
                results,
                {**{"flores_uk": MAIN_BENCHMARKS["flores_uk"]}, **FLORES_BENCHMARKS},
            )
            flores_df = calculate_average_rank(flores_df)

            long_flores_df = create_dataframe(
                results,
                {
                    **{"long_flores_uk": MAIN_BENCHMARKS["long_flores_uk"]},
                    **LONG_FLORES_BENCHMARKS,
                },
            )
            long_flores_df = calculate_average_rank(long_flores_df)

            # Update dropdown choices
            model_list = main_df["Model"].tolist() if "Model" in main_df.columns else []

            # Return updated table and dropdown
            return (
                main_df.sort_values("Average Rank", ascending=True),
                gr.Dropdown.update(choices=model_list),
            )

        # Connect event handlers
        main_sort_by.change(
            fn=update_main_table,
            inputs=[main_sort_by, main_sort_asc, main_show_relative],
            outputs=main_table,
        )

        main_sort_asc.change(
            fn=update_main_table,
            inputs=[main_sort_by, main_sort_asc, main_show_relative],
            outputs=main_table,
        )

        main_show_relative.change(
            fn=update_main_table,
            inputs=[main_sort_by, main_sort_asc, main_show_relative],
            outputs=main_table,
        )

        refresh_btn.click(
            fn=refresh_data,
            inputs=[],
            outputs=[main_table, model_selector],
        )

        benchmark_category.change(
            fn=update_detailed_table,
            inputs=[
                benchmark_category,
                detailed_sort_by,
                detailed_sort_asc,
                remove_dupes,
            ],
            outputs=detailed_table,
        )

        detailed_sort_by.change(
            fn=update_detailed_table,
            inputs=[
                benchmark_category,
                detailed_sort_by,
                detailed_sort_asc,
                remove_dupes,
            ],
            outputs=detailed_table,
        )

        detailed_sort_asc.change(
            fn=update_detailed_table,
            inputs=[
                benchmark_category,
                detailed_sort_by,
                detailed_sort_asc,
                remove_dupes,
            ],
            outputs=detailed_table,
        )

        remove_dupes.change(
            fn=update_detailed_table,
            inputs=[
                benchmark_category,
                detailed_sort_by,
                detailed_sort_asc,
                remove_dupes,
            ],
            outputs=detailed_table,
        )

        compare_btn.click(
            fn=update_comparison,
            inputs=[model_selector],
            outputs=[comparison_chart, comparison_table],
        )

        viz_btn.click(
            fn=update_visualization, inputs=[metric_selector], outputs=[metric_plot]
        )

        with open("README.md", "r", encoding="utf-8") as f:
            readme_content = f.read()
            readme_content = readme_content[readme_content.find("---", 5):]


        # Footer
        gr.Markdown(
            readme_content
        )


    return app


# Run the app when the script is executed
if __name__ == "__main__":
    app = create_leaderboard_app()
    app.launch()
