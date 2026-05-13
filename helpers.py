import yaml
import math

with open("config.yaml", "r") as file:
    load_config = yaml.safe_load(file)

def NoCarsToCongestionLevels(count):
    bins = load_config.get("environment", {}).get("bins", {})
    if not bins: return None
    
    for key, value in bins.items():
        binName = key.get()
        binRangeEnd = key[binName]["end"] or math.inf
        if(count <= binRangeEnd): return binName
        
    return None

