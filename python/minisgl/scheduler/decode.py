from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    running_reqs: Set[Req] = field(default_factory=set)
    """
    
    如果在decode的过程中，新token命中终止token了，那么在scheduler的self.decode_manager.remove_req(req)这个时候
    就会调用remove_req把req从running_reqs里移除掉，这样就不会再被scheduler调度了。
    如果没有结束，那么他就会一直decode直到结束或者被用户取消（目前没有取消的接口）。
    """

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    @property
    def inflight_tokens(self) -> int:
        return sum(req.remain_len for req in self.running_reqs)

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        return Batch(reqs=list(self.running_reqs), phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
