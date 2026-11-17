#加载toml配置文件内容
import tomllib
from pathlib import Path

def load_tool(name):
    pyproj_path=Path(__file__).parent.parent.parent / "pyproject.toml"

    with open(pyproj_path,"rb") as file:
        data=tomllib.load(file)

    tool=None
    if name=="processor":
        tool=data.get("tool",{}).get("processor",{})
    elif name=="word2vec_package":
        tool=data.get("tool",{}).get("word2vec_package",{})
    else:
        raise KeyError(f"unknown tool: {name}")

    return tool
