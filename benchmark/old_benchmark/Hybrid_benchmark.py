"""
Hybrid benchmark settings (parallel to Drain_benchmark.py),
used to hardcode the model checkpoint per dataset.

For each dataset, set 'checkpoint_dir' to the absolute directory
that contains model.pt and config.json. Optional fields:
 - device: 'cpu' or 'cuda'
 - min_match_prob: float threshold for matching

Example (Windows path):
  r"C:\\Users\\<you>\\...\\results\\actual\\...\\fold_0_Apache"
"""
BASE_DIR = r"C:\Users\schoe\Desktop\master\test_code\master\results\actual"
GRID_FOLDER = r"\nested_lodo_grid_1757385547"
CONFIG = r"\inner\batch_size=4096,bidirectional=True,device=cuda,early_stop=100,embed_dim=2,epochs=5,hidden_size=25,layers=3,lr=0.001,model=lstm,seed=42,use_projection=False,weight_decay=0.0001"


hybrid_settings = {
    "Proxifier": {
        "checkpoint_dir": BASE_DIR + GRID_FOLDER + r"\outer_10_Proxifier" + CONFIG + r"\fold_6_Mac",
        "device": "cpu",
        "min_match_prob": 0.5,
    },
    "Linux": {
        "checkpoint_dir": BASE_DIR + GRID_FOLDER + r"\outer_6_Linux" + CONFIG + r"\fold_6_Mac",
        "device": "cpu",
        "min_match_prob": 0.5,
    },
    "Apache": {
        "checkpoint_dir": BASE_DIR + GRID_FOLDER + r"\outer_0_Apache" + CONFIG + r"\fold_6_Mac",
        "device": "cpu",
        "min_match_prob": 0.5,
    },
    "Zookeeper": {
        "checkpoint_dir": BASE_DIR + GRID_FOLDER + r"\outer_13_Zookeeper" + CONFIG + r"\fold_6_Mac",
        "device": "cpu",
        "min_match_prob": 0.5,
    },
    "Hadoop": {
        "checkpoint_dir": BASE_DIR + GRID_FOLDER + r"\outer_4_Hadoop" + CONFIG + r"\fold_6_Mac",
        "device": "cpu",
        "min_match_prob": 0.5,
    },
    "HealthApp": {
        "checkpoint_dir": BASE_DIR + GRID_FOLDER + r"\outer_5_HealthApp" + CONFIG + r"\fold_6_Mac",
        "device": "cpu",
        "min_match_prob": 0.5,
    },
    "OpenStack": {
        "checkpoint_dir": BASE_DIR + GRID_FOLDER + r"\outer_9_OpenStack" + CONFIG + r"\fold_6_Mac",
        "device": "cpu",
        "min_match_prob": 0.5,
    },
    "HPC": {
        "checkpoint_dir": BASE_DIR + GRID_FOLDER + r"\outer_3_HPC" + CONFIG + r"\fold_6_Mac",
        "device": "cpu",
        "min_match_prob": 0.5,
    },
    "Mac": {
        "checkpoint_dir": BASE_DIR + GRID_FOLDER + r"\outer_7_Mac" + CONFIG + r"\fold_9_Proxifier",
        "device": "cpu",
        "min_match_prob": 0.5,
    },
    "OpenSSH": {
        "checkpoint_dir": BASE_DIR + GRID_FOLDER + r"\outer_8_OpenSSH" + CONFIG + r"\fold_6_Mac",
        "device": "cpu",
        "min_match_prob": 0.5,
    },
    "Spark": {
        "checkpoint_dir": BASE_DIR + GRID_FOLDER + r"\outer_11_Spark" + CONFIG + r"\fold_6_Mac",
        "device": "cpu",
        "min_match_prob": 0.5,
    },
    "Thunderbird": {
        "checkpoint_dir": BASE_DIR + GRID_FOLDER + r"\outer_12_Thunderbird" + CONFIG + r"\fold_6_Mac",
        "device": "cpu",
        "min_match_prob": 0.5,
    },
    "BGL": {
        "checkpoint_dir": BASE_DIR + GRID_FOLDER + r"\outer_1_BGL" + CONFIG + r"\fold_6_Mac",
        "device": "cpu",
        "min_match_prob": 0.5,
    },
    "HDFS": {
        "checkpoint_dir": BASE_DIR + GRID_FOLDER + r"\outer_2_HDFS" + CONFIG + r"\fold_6_Mac",
        "device": "cpu",
        "min_match_prob": 0.5,
    },
}
