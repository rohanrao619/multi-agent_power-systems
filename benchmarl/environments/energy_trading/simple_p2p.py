#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    data_path: str = MISSING
    dt: float = MISSING
    eps_len: int = MISSING
    es_P: float = MISSING
    es_capacity: list[float] = MISSING
    es_efficiency: list[float] = MISSING
    ToU: list[float] = MISSING
    FiT: float = MISSING
    use_contracts: bool = MISSING
    max_contract_qnt: float = MISSING
    use_single_group: bool = MISSING