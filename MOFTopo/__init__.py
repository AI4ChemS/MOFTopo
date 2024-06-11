from os import chdir
from pathlib import Path

from MOFTopo.common.env_utils import load_envs, get_env


load_envs()
PROJECT_ROOT: Path = Path(get_env("PROJECT_ROOT"))
assert (
    PROJECT_ROOT.exists()
), "You must configure the PROJECT_ROOT environment variable in a .env file!"

chdir(PROJECT_ROOT)