# test_dummy_run.py
import sys
from pathlib import Path
from unittest import mock
import asyncio
import random
import json

import cot_pipeline  # the pipeline script from the canvas
import prompt_gen

###############################################################################
# Dummy model & fake batch ----------------------------------------------------
###############################################################################

class DummyModel:
    """Stand-in returned by configure_gemini()."""
    pass


# async def fake_batch(prompts, model):
#     """Return synthetic answers that look ‘correct’ for each dataset."""
#     outs = []
#     for p in prompts:
#         # Boxed-number tasks
#         if "\\boxed" in p:
#             outs.append("… therefore the answer is \\boxed{64}")
#         # Multiple-choice tasks
#         elif "best answer is" in p.lower():
#             outs.append("The best answer is A")
#         # True/False/Neither
#         elif "True, False, or Neither" in p:
#             outs.append("Final answer: True.")
#         # Yes/No
#         elif "Yes or No" in p:
#             outs.append("No.")
#         else:
#             outs.append("Placeholder.")
#     return outs

async def fake_batch(prompt, model):
    """Return synthetic answers that look ‘correct’ for each dataset."""
    outs = []
    p = prompt
    # Boxed-number tasks
    if "\\boxed" in p:
        outs.append("… therefore the answer is \\boxed{64}")
    # Multiple-choice tasks
    elif "best answer is" in p.lower():
        outs.append("The best answer is A")
    # True/False/Neither
    elif "True, False, or Neither" in p:
        outs.append("Final answer: True.")
    # Yes/No
    elif "Yes or No" in p:
        outs.append("No.")
    else:
        outs.append("Placeholder.")
        
    return outs[0]

###############################################################################
# Offline runner --------------------------------------------------------------
###############################################################################

def offline_main(data_root="."):
    """Run cot_pipeline.main() without network calls."""
    with  mock.patch.object(prompt_gen, "batch_call_gemini_api", fake_batch):

        # Backup CLI args and replace
        argv_backup = sys.argv[:]
        sys.argv = ["prompt_gen.py", "--root", data_root, "--n", "3"]

        try:
            prompt_gen.main()      # executes with the patched functions
        finally:
            sys.argv = argv_backup   # restore original argv


if __name__ == "__main__":
    offline_main("./data/")   # path to your dataset root