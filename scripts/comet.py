import sys
import comet_ml

exp = comet_ml.start(
    workspace="mastns",
    project_name="thesis-simos",
)

if "--task" in sys.argv:
    task_name = sys.argv[sys.argv.index("--task") + 1]
    exp.set_name(task_name)

import h1_2_humanoid  # noqa: E402, F401
from isaaclab.cli import cli  # noqa: E402

sys.exit(cli())