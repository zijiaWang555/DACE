import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type=str, default='FACE')
parser.add_argument('--dataset', type=str, default='JOB-light')
parser.add_argument('--alpha', type=float, default=0.1)
args = parser.parse_args()

import Backbones.FACE.kd
import Backbones.UAE.kd
import Backbones.NeuroCard.kd
import Backbones.ALECE.kd
if args.backbone == 'FACE':
    Backbones.FACE.kd.run(args)
elif args.backbone == 'NeuroCard':
    Backbones.NeuroCard.kd.run(args)
elif args.backbone == 'UAE':
    Backbones.UAE.kd.run(args)
elif args.backbone == 'ALECE':
    #The configuration file in arg_parser.py
    Backbones.ALECE.kd.run()
else:
    print("Error Backbone")


