import torch
import ldm_patched.modules.model_management as model_management

from modules.config import path_clip_vision
from ldm_patched.modules.model_patcher import ModelPatcher
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class Interrogator:
    def __init__(self):
        self.blip_model = None
        self.blip_processor = None
        self.load_device = torch.device('cpu')
        self.offload_device = torch.device('cpu')
        self.dtype = torch.float32

    @torch.no_grad()
    @torch.inference_mode()
    def interrogate(self, img_rgb):
        if self.blip_model is None:
            self.blip_processor = Blip2Processor.from_pretrained(
                "Salesforce/blip2-opt-6.7b", cache_dir=path_clip_vision)
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-6.7b", cache_dir=path_clip_vision)
            model.eval()

            self.load_device = model_management.text_encoder_device()
            self.offload_device = model_management.text_encoder_offload_device()
            self.dtype = torch.float32

            model.to(self.offload_device)

            if model_management.should_use_fp16(device=self.load_device):
                model.half()
                self.dtype = torch.float16

            self.blip_model = ModelPatcher(model, load_device=self.load_device, offload_device=self.offload_device)

        model_management.load_model_gpu(self.blip_model)

        inputs = self.blip_processor(images=img_rgb, return_tensors="pt").to(self.load_device, self.dtype)
        generated_ids = self.blip_model.model.generate(**inputs, max_new_tokens=75)
        caption = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Move BLIP-2 model back to CPU to free GPU memory. The model remains
        # initialized but is offloaded until the next interrogation call loads
        # it again. This avoids hogging VRAM when captioning is done.
        loaded_blip = model_management.LoadedModel(self.blip_model)
        if loaded_blip in model_management.current_loaded_models:
            idx = model_management.current_loaded_models.index(loaded_blip)
            model_management.current_loaded_models.pop(idx).model_unload()
            model_management.soft_empty_cache()

        return caption


default_interrogator = Interrogator().interrogate
