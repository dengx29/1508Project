# Main script to run the search benchmark

import argparse
import json

from src.benchmark_runner import load_config, run_full_benchmark


def main():
    # Setup command line options
    parser = argparse.ArgumentParser(description="Run the retrieval benchmark with a YAML config.")
    parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    # Load the settings
    config = load_config(args.config)
    print("Config loaded.")
    print("\n=== Running Benchmark ===")
    
    # Run the whole test
    result = run_full_benchmark(config)
    chunks = result["chunks"]
    sampled_queries = result["sampled_queries"]
    summary = result["summary"]
    log_path = result["log_path"]

    # Print data counts
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Sampled queries: {len(sampled_queries)}")
    print(f"    single-entity: {sum(1 for q in sampled_queries if q['entity_group'] == 'single-entity')}")
    print(f"    multi-entity:  {sum(1 for q in sampled_queries if q['entity_group'] == 'multi-entity')}")
    print(f"  JSON log saved to: {log_path}")

    # Show score table
    print("\n=== 5. Evaluation ===")
    print(summary.to_string(index=False))

    # Show chart file paths
    print("\n=== 6. Visualization ===")
    for name, path in result["output_paths"].items():
        print(f"  {name}: {path}")

    # Show time and memory use
    print("\n=== 7. Profiling Summary ===")
    with open(log_path, encoding="utf-8") as f:
        log = json.load(f)
    for stage, info in log["stages"].items():
        vram = f", VRAM peak: {info['peak_vram_bytes']/1e9:.2f} GB" if info.get("peak_vram_bytes") else ""
        print(f"  {stage}: {info['duration_seconds']:.2f}s, RSS: {info['rss_bytes']/1e9:.2f} GB{vram}")
    
    # Show index file sizes
    for label, size in log["disk_sizes"].items():
        print(f"  {label}: {size/1e6:.1f} MB")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()