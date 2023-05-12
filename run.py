import asyncio
import queue
import argparse
from scripts.model import main

def run_model(days, model, lease, sale):
    if model:
        if sale and lease:
            main(days, None)
        else:
            if lease:
                main(days, 'lease')
            if sale:
                main(days, 'sale')

    else:
        print("Create model was not specified")

parser  = argparse.ArgumentParser(description = 'Enter the amount of days to retrieve information from')

parser.add_argument("days", help='A natural number for the number of days to retrieve information from', metavar='days')
parser.add_argument("-l", "--lease", action="store_true", help='specify if you want to generate a new model specifically for lease data \'-m must be specified\'')
parser.add_argument("-s", "--sale", action="store_true", help='specify if you want to generate a new model specifically for sale data \'-m must be specified\'')
parser.add_argument("-m", "--model", action = "store_true", help='specify if you want to generate a model')

args = parser.parse_args()

print(args.days)
print(args.lease)
print(args.sale)
print(args.model)


try:
    days = int(args.days)
except:
    print("This cannot be converted into an int, please enter a different value")

run_model(days, args.model, args.lease, args.sale)