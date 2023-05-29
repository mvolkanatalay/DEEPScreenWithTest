import argparse
from test_DEEPScreen import test_DEEPScreen 


parser = argparse.ArgumentParser(description='DEEPScreen arguments')

parser.add_argument(
    '--targetid',
    type=str,
    default="CHEMBL286",
    metavar='TID',
    help='Target ChEMBL ID')

parser.add_argument(
    '--modelfile',
    type=str,
    default="",
    metavar='MF',
    help='Model file name')

parser.add_argument(
    '--testfile',
    type=str,
    default="",
    metavar='TF',
    help='Test file name')

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    test_DEEPScreen(args.targetid, args.modelfile, args.testfile)

