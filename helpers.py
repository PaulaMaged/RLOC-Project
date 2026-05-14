import yaml
import math
from pathlib import Path
import pickle
from glom import glom
from glom import assign

def NoCarsToCongestionLevels(count):
    
    bins = getConfigData().get("environment", {}).get("bins", {})
    if not bins: return None
    
    for key, value in bins.items():
        binName = key.get()
        binRangeEnd = key[binName]["end"] or math.inf
        if(count <= binRangeEnd): return binName
        
    return None

def is_tuple_of_ints(obj):
    return isinstance(obj, tuple) and all(isinstance(x, int) for x in obj)

def getConfigData(fileName = "config.yaml", keys = []):
    file_path = Path(fileName)
    file_path.touch(exist_ok=True)

    with file_path.open("r") as file:
        config = yaml.safe_load(file) or {}
    
    if config and keys:
        filtered_dict = {}
        for key in keys:
            filtered_dict[key] = glom(config, key, defualt=None)
        return filtered_dict
    else:
        return config
    
# def persistData(overwrite=False, data=None):
#     if not data:
#         return
        
#     folder = Path("db")
#     folder.mkdir(exist_ok=True)
#     identifier = getConfigData(keys=["persistence.identifier"])[f"persistence.${type(data).__name__}.identifier"]
    
#     if not identifier:
#         updateConfigData({identifier: 0})
#     else:
#         identifier += 1
        
#     if not cls:

#     else:
        
    
# def getPersistedData(path = None, keys = []):
    
# setup folder path and filename to save agent and environment for sharing and cummulative learning

# with identifer_file_path.open("r") as f:
#     if()
#     content = f.read()
#     if(identifer_file_path.stat().st_size != 0):
#         identifier = content.split(":")[-1].
#         identifier += 1

# with identifer_file_path.open("w") as f:
#     f.write("")
    