import argparse
import math
import pandas as pd

def sigmoid(x):
    return 1/(1+ math.exp(-x)) #sigmoid function
def main_program(tsv_file):
    column_names = ['read_id', 'ctg', 'pos', 'prob']
    df = pd.read_csv(tsv_file, delimiter='\t', names=column_names)
    df['prob'] = df['prob'].apply(sigmoid)
    df.to_csv('modified.tsv', sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv_file', type=str)
    args = parser.parse_args()
    main_program(args.tsv_file)
