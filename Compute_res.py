import argparse
import sys
import logging

from config import read_config
from utils import LoggingLevel

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='AnoRocket training')
    parser.add_argument('-c', '--config', required=True, help='config file path')
    return parser.parse_args(args)

def train(opts=None):
    # parse arguments
    if opts is None:
        args = sys.argv[1:]        
        args = parse_args(args)
        opts = read_config(args.config)

    # set Logging Level
    level = LoggingLevel(opts['miscellaneous']['log_level'])
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d - [%(levelname)s] - %(message)s',
        datefmt='%d %b %H:%M:%S',
        level=level
        )

   
        
if __name__ == "__main__":
    train()