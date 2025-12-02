import argparse
from pathlib import Path

from src.config import QueryGenConfig
from src.query_gen import QueryGenerator
from src.data import load_beir_dataset, save_query_bank, beir_queries_to_synthetic
from src.utils import get_device


def generate(
    dataset_name: str,
    device: str,
    num_single: int = 1000,
    num_multi: int = 200,
    num_adversarial: int = 200
):
    device = get_device(device)
    print(f"Using device: {device}")
    
    corpus, queries, qrels = load_beir_dataset(dataset_name)
    
    beir_queries = beir_queries_to_synthetic(queries, qrels)
    
    config = QueryGenConfig()
    if device == "mps":
        model_name = config.model_name_mps
    else:
        model_name = config.model_name
    
    generator = QueryGenerator(
        model_name=model_name,
        device=device,
        queries_per_doc=config.queries_per_doc
    )
    
    synthetic_queries = generator.generate_query_bank(
        corpus,
        num_single=num_single,
        num_multi=num_multi,
        num_adversarial=num_adversarial
    )
    
    all_queries = beir_queries + synthetic_queries
    
    output_path = Path("data") / dataset_name / "query_bank.json"
    save_query_bank(all_queries, output_path)
    
    print(f"Generated {len(all_queries)} queries")
    print(f"  BEIR queries: {len(beir_queries)}")
    print(f"  Synthetic queries: {len(synthetic_queries)}")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nfcorpus")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-single", type=int, default=1000)
    parser.add_argument("--num-multi", type=int, default=200)
    parser.add_argument("--num-adversarial", type=int, default=200)
    args = parser.parse_args()
    
    generate(
        dataset_name=args.dataset,
        device=args.device,
        num_single=args.num_single,
        num_multi=args.num_multi,
        num_adversarial=args.num_adversarial
    )


if __name__ == "__main__":
    main()

