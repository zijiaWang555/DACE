import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type=str, default='FACE')
parser.add_argument('--dataset', type=str, default='JOB_light')
args = parser.parse_args()

import Backbones.FACE.dt
import Backbones.ALECE.dt
import Backbones.UAE.dt
import Backbones.NeuroCard.dt
if args.backbone == 'FACE':
    Backbones.FACE.dt.run(args)
elif args.backbone == 'NeuroCard':
    Backbones.NeuroCard.dt.run(args)
elif args.backbone == 'UAE':
    Backbones.Uae.dt.run(args)
elif args.backbone == 'ALECE':
    #The configuration file in arg_parser.py
      Backbones.ALECE.dt.run()
else:
    print("Error Backbone")


