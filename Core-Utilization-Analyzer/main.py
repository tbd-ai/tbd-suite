
import argparse
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("file", help="CSV file to analyze")
args = parser.parse_args()

# fields we benchmark (usage metrics collected)
fields = {
    'fma': 'sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed',
    'fp16': 'sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed',
    'tensor': 'sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed',
    'sfu': 'sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed'
}

# stores the numerator of the utilizations
field_cycles_total = dict()
for x in fields: field_cycles_total[x] = 0

# denominator of utilization
cycles_total = 0

# read the csv file
d = pd.read_csv(args.file)
max_id = max(d['ID'])

for cur_id in tqdm(range(max_id), leave=False):
    sec = d[d['ID'] == cur_id]

    cycles = int(sec[sec['Metric Name']=='gpc__cycles_elapsed.max']['Metric Value'])
    cycles_total += cycles

    for field in fields:
        field_cycles_total[field] += cycles * float(sec[sec['Metric Name']==fields[field]]['Metric Value']) / 100

for field in fields:
    util = field_cycles_total[field] / cycles_total
    print("%s\t%.5f %%" % (field, util*100))
