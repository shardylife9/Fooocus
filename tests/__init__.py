import sys
import pathlib
import types

sys.path.append(pathlib.Path(f"{__file__}/../modules").parent.resolve())

# Provide very small stubs for optional dependencies so that the tests can run
# in environments where the real libraries are unavailable.
try:  # pragma: no cover - only executed when dependencies are missing
    import numpy # type: ignore
except ModuleNotFoundError:  # pragma: no cover - minimal stub
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.uint8 = 0
    class ndarray:
        pass
    numpy_stub.ndarray = ndarray
    sys.modules["numpy"] = numpy_stub

try:  # pragma: no cover - only executed when dependencies are missing
    import PIL # type: ignore
except ModuleNotFoundError:  # pragma: no cover - minimal stub
    pil_stub = types.ModuleType("PIL")
    class StubImage:
        class Resampling:
            LANCZOS = 1
        LANCZOS = 1
    pil_stub.Image = StubImage
    sys.modules["PIL"] = pil_stub
    sys.modules["PIL.Image"] = StubImage

try:  # pragma: no cover - only executed when dependencies are missing
    import torch # type: ignore
except ModuleNotFoundError:  # pragma: no cover - minimal stub
    torch_stub = types.ModuleType("torch")
    class DummyTensor:
        def __init__(self, *args, **kwargs):
            pass
        def to(self, *args, **kwargs):
            return self

    class device(str):
        pass

    def tensor(*args, **kwargs):
        return DummyTensor()

    def no_grad():
        def wrapper(fn):
            return fn
        return wrapper

    def inference_mode():
        def wrapper(fn):
            return fn
        return wrapper

    torch_stub.tensor = tensor
    torch_stub.device = device
    torch_stub.no_grad = no_grad
    torch_stub.inference_mode = inference_mode
    torch_stub.float32 = "float32"
    torch_stub.float16 = "float16"
    sys.modules["torch"] = torch_stub

try:  # pragma: no cover - only executed when dependencies are missing
    import cv2 # type: ignore
except ModuleNotFoundError:  # pragma: no cover - minimal stub
    sys.modules["cv2"] = types.ModuleType("cv2")

try:  # pragma: no cover - only executed when dependencies are missing
    import psutil # type: ignore
except ModuleNotFoundError:  # pragma: no cover - minimal stub
    sys.modules["psutil"] = types.ModuleType("psutil")

try:  # pragma: no cover - only executed when dependencies are missing
    import safetensors.torch # type: ignore
except ModuleNotFoundError:  # pragma: no cover - minimal stub
    safetensors_stub = types.ModuleType("safetensors")
    safetensors_torch_stub = types.ModuleType("safetensors.torch")
    safetensors_stub.torch = safetensors_torch_stub
    sys.modules["safetensors"] = safetensors_stub
    sys.modules["safetensors.torch"] = safetensors_torch_stub

# Stub out parts of the ldm_patched package that are imported during tests.
ldm_patched_stub = types.ModuleType("ldm_patched")
ldm_modules_stub = types.ModuleType("ldm_patched.modules")
model_management_stub = types.ModuleType("ldm_patched.modules.model_management")
model_management_stub.text_encoder_device = lambda: torch.device("cpu")
model_management_stub.text_encoder_offload_device = lambda: torch.device("cpu")
model_management_stub.should_use_fp16 = lambda *args, **kwargs: False
model_management_stub.load_model_gpu = lambda *args, **kwargs: None
model_patcher_stub = types.ModuleType("ldm_patched.modules.model_patcher")
class DummyModelPatcher:
    def __init__(self, model, load_device=None, offload_device=None):
        self.model = model
model_patcher_stub.ModelPatcher = DummyModelPatcher

ldm_modules_stub.model_management = model_management_stub
ldm_modules_stub.model_patcher = model_patcher_stub
args_parser_stub = types.ModuleType("ldm_patched.modules.args_parser")
class DummyParser:
    def __init__(self):
        self.defaults = {}
    def add_argument(self, *args, **kwargs):
        pass
    def set_defaults(self, **kwargs):
        self.defaults.update(kwargs)
    def parse_args(self):
        base = dict(disable_offload_from_vram=False,
                    disable_analytics=False,
                    disable_in_browser=False,
                    in_browser=True,
                    port=None,
                    preset=None,
                    output_path=None,
                    temp_path=None)
        base.update(self.defaults)
        return types.SimpleNamespace(**base)
parser = DummyParser()
args_parser_stub.parser = parser
ldm_modules_stub.args_parser = args_parser_stub
ldm_patched_stub.modules = ldm_modules_stub

sys.modules.setdefault("ldm_patched", ldm_patched_stub)
sys.modules.setdefault("ldm_patched.modules", ldm_modules_stub)
sys.modules.setdefault("ldm_patched.modules.model_management", model_management_stub)
sys.modules.setdefault("ldm_patched.modules.model_patcher", model_patcher_stub)
sys.modules.setdefault("ldm_patched.modules.args_parser", args_parser_stub)

try:  # pragma: no cover - only executed when dependencies are missing
    import transformers # type: ignore
except ModuleNotFoundError:  # pragma: no cover - minimal stub
    transformers_stub = types.ModuleType("transformers")
    class DummyProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return None
    class DummyModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return None
    transformers_stub.Blip2Processor = DummyProcessor
    transformers_stub.Blip2ForConditionalGeneration = DummyModel
    sys.modules["transformers"] = transformers_stub
