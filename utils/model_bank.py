import json
import os
from datetime import datetime

# model_bank/champion.json points at the model whose predictions are used in production.
# Deploying (promoting) a model = an approved edit of this file between monthly runs;
# the previous model stays in model_bank for one-step rollback.
MODEL_BANK_DIRECTORY = "model_bank/"
CHAMPION_FILEPATH = MODEL_BANK_DIRECTORY + "champion.json"


def read_champion():
    if not os.path.exists(CHAMPION_FILEPATH):
        return None
    with open(CHAMPION_FILEPATH) as file:
        return json.load(file)


def get_role(modelname):
    champion = read_champion()
    return "champion" if champion is not None and champion["champion"] == modelname else "challenger"


def write_champion(champion_modelname, challenger_modelnames, reason):
    pointer = {
        "champion": champion_modelname,
        "challengers": challenger_modelnames,
        "promoted_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reason": reason,
    }
    with open(CHAMPION_FILEPATH, "w") as file:
        json.dump(pointer, file, indent=2)
    return pointer
