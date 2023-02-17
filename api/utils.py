# Copyright (C) 2023 KNS Group LLC (YADRO)
# SPDX-License-Identifier: Apache-2.0


import os
import numpy as np


def get_model_path(model_dir: str, launcher_type: str):
    full_path = os.path.join(model_dir, launcher_type)
    return full_path
