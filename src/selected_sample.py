import argparse


def main(args):
    with open(args.question_only_path, "r") as qf, open(args.retrieved_path, "r") as rf:
        question_only_correct = [line.strip() for line in qf.readlines()]
        retrieved_correct = [line.strip() for line in rf.readlines()]
        intersection = set(question_only_correct).intersection(retrieved_correct)
        result = set(retrieved_correct).difference(intersection)
        print("Len that retrieved correct but question not: ", len(result))    
    
    
    with open(args.output_path, "w") as f:
        f.write("\n".join(result))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_only_path", type=str)
    parser.add_argument("--retrieved_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    main(args)