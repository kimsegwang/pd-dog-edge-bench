"""두 모델이 같은 규칙으로 onnxruntime 세션을 만들도록 하는 헬퍼."""
from pathlib import Path

import onnxruntime as ort

_OPT = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def make_session(model_path: Path, ort_cfg: dict, intra_op_threads: int | None = None) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(intra_op_threads if intra_op_threads is not None else ort_cfg["intra_op_threads"])
    so.inter_op_num_threads = int(ort_cfg.get("inter_op_threads", 1))
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = _OPT[ort_cfg.get("graph_optimization", "all")]
    if not ort_cfg.get("allow_spinning", True):
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
