from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List, Optional, Union

import numpy as np


def _default_model_path() -> str:
    """返回仓库内置 Qwen3-0.6B 模型的默认路径。"""

    repo_root = Path(__file__).resolve().parents[1]
    local_model = repo_root / "models" / "Qwen3-0.6B"
    return str(local_model)


class Embedder:

    def __init__(
        self,
        model_name: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        device_map: Optional[str] = None,
        batch_size: int = 8,
        max_length: int = 1024,
        normalize: bool = True,
    ) -> None:
        """初始化句向量编码器。

        Args:
            model_name: HuggingFace/ModelScope 模型名称或本地路径。
                默认指向仓库自带的 `models/Qwen3-0.6B` 目录。
            device: 明确指定推理设备；默认自动选择 GPU/CPU。
            batch_size: 编码批大小。
            max_length: 句子分词后的最大序列长度。
            normalize: 是否对输出向量进行 L2 归一化。
        """

        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        self._dim: Optional[int] = None
        self._target_device = None
        self._device_map = device_map

        self._mode = "hf"
        if model_name is None:
            model_name = _default_model_path()
        else:
            model_name = str(model_name)

        if str(model_name).lower() == "dummy":
            self._mode = "dummy"
            self.device = "cpu"
            self.qwen_tokenizer = None
            self.qwen_model = None
            self._dim = 64
            return

        try:
            import torch  # noqa: F401 仅检测可用性

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    
        if isinstance(device, str) and "," in device:
            if device_map is None:
                raise ValueError(
                    "检测到多个 device，但未指定 device_map。请使用 --device-map auto 或分配单个设备"
                )
        self.device = device

        from transformers import AutoModel, AutoTokenizer
        import torch
        import importlib.util

        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        device_is_cuda = isinstance(self.device, str) and self.device.startswith("cuda")
        torch_dtype = torch.float16 if device_is_cuda else torch.float32
        has_accelerate = importlib.util.find_spec("accelerate") is not None
        device_map = self._device_map
        if device_map is None and device_is_cuda and has_accelerate:
            device_map = "auto"

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
        }
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        self.qwen_model = AutoModel.from_pretrained(model_name, **model_kwargs)
        if device_map is None:
            self.qwen_model.to(self.device)
            self._target_device = torch.device(self.device)
        else:
            if hasattr(self.qwen_model, "device"):
                self._target_device = torch.device(self.qwen_model.device)
            else:
                self._target_device = next(self.qwen_model.parameters()).device

        self._dim = int(getattr(self.qwen_model.config, "hidden_size"))

    @property
    def dim(self) -> int:
        """
        获取当前使用的嵌入向量的维度大小。
        """
        return int(self._dim)

    def encode(self, sentences: Iterable[str]) -> List[List[float]]:
        """
        将一批句子编码为向量嵌入。

        Args:
            sentences (Iterable[str]): 要编码的句子集合。

        Returns:
            List[List[float]]: 每个句子对应的向量嵌入列表。

        Raises:
            RuntimeError: 如果 Qwen 模型未正确初始化。
        """
        sents = list(sentences)
        # 如果输入为空，则返回空列表
        if not sents:
            return []

        if self._mode == "dummy":
            vectors: List[List[float]] = []
            for text in sents:
                h = hashlib.sha1(text.encode("utf-8", errors="ignore")).digest()
                seed = int.from_bytes(h[:8], byteorder="little", signed=False)
                rng = np.random.default_rng(seed)
                arr = rng.normal(size=self._dim).astype(np.float32)
                if self.normalize:
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        arr = arr / norm
                vectors.append(arr.tolist())
            return vectors

        if self.qwen_model is None:
            raise RuntimeError("Qwen 模型未正确初始化")

        import torch

        self.qwen_model.eval()
        outputs: List[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(sents), self.batch_size):
                batch = sents[i : i + self.batch_size]
                toks = self.qwen_tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                target_device = self._target_device
                if target_device is None:
                    if hasattr(self.qwen_model, "device"):
                        target_device = torch.device(self.qwen_model.device)
                    else:
                        target_device = next(self.qwen_model.parameters()).device
                toks = {k: v.to(target_device) for k, v in toks.items()}
                hidden = self.qwen_model(**toks).last_hidden_state
                mask = toks["attention_mask"].unsqueeze(-1)
                masked = hidden * mask
                summed = masked.sum(dim=1)
                lengths = mask.sum(dim=1).clamp(min=1)
                emb = summed / lengths
                if self.normalize:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                outputs.append(emb.cpu().float().numpy())

        arr = np.concatenate(outputs, axis=0)
        return arr.astype(np.float32).tolist()
