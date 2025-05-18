import unittest
from unittest import mock
import torch

from extras import interrogate


class TestInterrogate(unittest.TestCase):
    def test_default_interrogator_returns_caption(self):
        img = object()

        with mock.patch("extras.interrogate.Blip2Processor.from_pretrained") as mock_processor_from_pretrained, \
             mock.patch("extras.interrogate.Blip2ForConditionalGeneration.from_pretrained") as mock_model_from_pretrained, \
             mock.patch("extras.interrogate.ModelPatcher") as mock_model_patcher, \
             mock.patch("extras.interrogate.model_management.load_model_gpu"), \
             mock.patch("extras.interrogate.model_management.text_encoder_device", return_value=torch.device("cpu")), \
             mock.patch("extras.interrogate.model_management.text_encoder_offload_device", return_value=torch.device("cpu")), \
             mock.patch("extras.interrogate.model_management.should_use_fp16", return_value=False):

            mock_processor = mock.Mock()

            def processor_call(images, return_tensors="pt"):
                class Dummy:
                    def to(self, device, dtype=None):
                        return {"input_ids": torch.tensor([[1]])}
                return Dummy()

            mock_processor.side_effect = processor_call
            mock_processor.batch_decode.return_value = ["dummy caption"]
            mock_processor_from_pretrained.return_value = mock_processor

            mock_model = mock.Mock()
            mock_model.generate.return_value = torch.tensor([[1]])
            mock_model.eval.return_value = None
            mock_model.to.return_value = None
            mock_model_from_pretrained.return_value = mock_model

            mock_model_patcher.return_value = mock.Mock(model=mock_model)

            caption = interrogate.default_interrogator(img)

        self.assertIsInstance(caption, str)
        self.assertTrue(len(caption) > 0)


if __name__ == "__main__":
    unittest.main()
