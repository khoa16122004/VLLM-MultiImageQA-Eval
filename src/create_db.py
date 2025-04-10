import argparse
from utils import init_encode_model
from retriever import Retriever
def main(args):
    encode_model, dim = init_encode_model(args.model_name_encode)
    retriever = Retriever(args.output_index_dir, encode_model, dim)
    
    # Extract feature
    retriever.extract_db(args.dataset_dir, args.database_dir)
    # Indexing
    retriever.create_database(args.database_dir, args.output_index_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_encode", type=str)
    parser.add_argument("--dataset_dir", type=str, default="../dataset/MRAG_corpus")
    parser.add_argument("--database_dir", type=str)
    parser.add_argument("--output_index_dir", type=str)
    args = parser.parse_args()
    
    main(args)