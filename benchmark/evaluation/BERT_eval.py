"""
This file is part of TA-Eval-Rep.
Copyright (C) 2022 University of Luxembourg

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import sys
import os

sys.path.append('../')

# Import benchmark settings from Drain's evaluation framework.
from old_benchmark.BERT_benchmark import benchmark_settings
# Import the BertLogParser class instead of Drain's LogParser.
from logparser.BERT import LogParser
from evaluation.utils.common import common_args
from evaluation.utils.evaluator_main import evaluator, prepare_results
from evaluation.utils.postprocess import post_average

# Define dataset lists for 2k and full log datasets.
datasets_2k = [
    "Proxifier",
    #"Linux",
    #"Apache",
    #"Zookeeper",
    #"Hadoop",
    #"HealthApp",
    #"OpenStack",
    #"HPC",
    #"Mac",
    #"OpenSSH",
    #"Spark",
    #"Thunderbird",
    #"BGL",
    #"HDFS",
]

datasets_full = [
    #"Proxifier",
    #"Linux",
    #"Apache",
    #"Zookeeper",
    #"Hadoop",
    #"HealthApp",
    #"OpenStack",
    #"HPC",
    #"Mac",
    #"OpenSSH",
    #"Spark",
    #"Thunderbird",
    #"BGL",
    #"HDFS",
]

if __name__ == "__main__":
    args = common_args()
    data_type = "full" if args.full_data else "2k"
    input_dir = f"../../{data_type}_dataset/"
    output_dir = f"../../result/result_BertLogParser_{data_type}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_file = prepare_results(
        output_dir=output_dir,
        otc=args.oracle_template_correction,
        complex=args.complex,
        frequent=args.frequent
    )

    datasets = datasets_full if args.full_data else datasets_2k

    for dataset in datasets:
        setting = benchmark_settings[dataset]
        # Adjust the log file name according to the dataset type.
        log_file = setting['log_file'].replace("_2k", f"_{data_type}")
        indir = os.path.join(input_dir, os.path.dirname(log_file))

        # If parsing result already exists, skip processing.
        #if os.path.exists(os.path.join(output_dir, f"{dataset}_{data_type}.log_structured.csv")):
        #    parser = None
        #    print("Parsing result exists for dataset:", dataset)
        #else:
        parser = LogParser

        print("Using log format:", setting['log_format'])
        # Run evaluator for the current dataset.
        evaluator(
            dataset=dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            log_file=log_file,
            LogParser=parser,
            param_dict={
                'log_format': setting['log_format'],
                'indir': indir,
                'outdir': output_dir,
                'rex': setting['regex'],
                'st': setting['st'],
                # Pass drift_threshold parameter specific to BertLogParser.
                'drift_threshold': 0.3
            },
            otc=args.oracle_template_correction,
            complex=args.complex,
            frequent=args.frequent,
            result_file=result_file
        )  # evaluator internally saves the results into a summary file.

    metric_file = os.path.join(output_dir, result_file)
    post_average(metric_file, f"BertLogParser_{data_type}_complex={args.complex}_frequent={args.frequent}", args.complex, args.frequent)
