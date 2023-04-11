import sys
import os
sys.path.append(os.path.abspath('/home/anirudha/COL707/doh_traffic_analysis/code/classification/'))

from utils.util import load_data

# take first argument as the path to the file
path = sys.argv[1]
load_data(path)