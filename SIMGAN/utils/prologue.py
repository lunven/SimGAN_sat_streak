import argparse
import os

parser = argparse.ArgumentParser(description='Satellite Streaks Detection')

parser.add_argument('--i', type=str)
parser.add_argument('--o', type = str)
parser.add_argument('--k', type = str)
parser.add_argument('--m', type = str)
parser.add_argument('--n', type = int)
parser.add_argument('--l', type = float)
parser.add_argument('--b', type = int)
parser.add_argument('--v',type = int)

#parser.add_argument('--learning_rate', type = float, default = 5e-4, help = 'Learning rate for the optimizer')
args = parser.parse_args()

def get_args():
    return args
