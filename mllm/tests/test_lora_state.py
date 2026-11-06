"""Checkpoint parameter selection without a DeepSpeed runtime."""
import unittest
import copy
import tempfile

import torch

from vtimellm.train.lora_state import collect_lora_state


class LoraStateTests(unittest.TestCase):
    def test_bias_modes_and_single_pass_parameter_iterator(self):
        params = {
            "layer1.bias": torch.tensor([1.]),
            "layer1.lora_A.default.weight": torch.tensor([2.]),
            "layer2.lora_B.default.weight": torch.tensor([3.]),
            "unrelated.bias": torch.tensor([4.]),
            "layer2.bias": torch.tensor([5.]),
            "layer2.weight": torch.tensor([6.]),
        }
        lora = {"layer1.lora_A.default.weight", "layer2.lora_B.default.weight"}
        for mode, expected in (
            ("none", lora),
            ("lora_only", lora | {"layer1.bias", "layer2.bias"}),
            ("all", lora | {"layer1.bias", "layer2.bias", "unrelated.bias"}),
        ):
            gathered = []

            def gather(value, name):
                gathered.append(name)
                return value.clone()

            result = collect_lora_state(iter(params.items()), mode, gather)
            self.assertEqual(set(result), expected)
            self.assertEqual(set(gathered), expected)
            for name in expected:
                torch.testing.assert_close(result[name], params[name])
                self.assertIsNot(result[name], params[name])

    def test_unknown_bias_mode_is_rejected(self):
        with self.assertRaises(NotImplementedError):
            collect_lora_state(iter(()), "unexpected", lambda value, name: value)

    def test_peft_wrapped_linear_bias_is_selected(self):
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2)),
            LoraConfig(target_modules=["0"], r=2, lora_alpha=4, bias="lora_only"),
        )
        result = collect_lora_state(model.named_parameters(), "lora_only", lambda value, name: value.detach().clone())
        self.assertEqual(set(result), {
            "base_model.model.0.base_layer.bias",
            "base_model.model.0.lora_A.default.weight",
            "base_model.model.0.lora_B.default.weight",
        })

    def test_trained_peft_bias_and_adapter_survive_repository_load(self):
        from peft import LoraConfig, get_peft_model
        from vtimellm.model.builder import load_lora

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))

            def forward(self, value):
                return self.model(value)

        torch.manual_seed(23)
        original = TinyModel()
        restored_base = copy.deepcopy(original)
        model = get_peft_model(original, LoraConfig(target_modules=["0"], r=2, lora_alpha=4, bias="lora_only"))
        with torch.no_grad():
            model.base_model.model.model[0].base_layer.bias.fill_(7.)
            model.base_model.model.model[0].lora_B.default.weight.fill_(.25)
        model.eval()
        inputs = torch.tensor([[.1, .2, .3]])
        with tempfile.TemporaryDirectory() as directory:
            state = collect_lora_state(model.named_parameters(), "lora_only", lambda value, name: value.detach().clone())
            non_lora = {name: value.detach().clone() for name, value in model.named_parameters()
                        if "lora_" not in name and value.requires_grad}
            model.save_pretrained(directory, state_dict=state)
            torch.save(non_lora, directory + "/non_lora_trainables.bin")
            restored = load_lora(restored_base, directory)
            restored.eval()
            torch.testing.assert_close(restored(inputs), model(inputs))
            torch.testing.assert_close(restored.base_model.model.model[0].base_layer.bias,
                                       model.base_model.model.model[0].base_layer.bias)


if __name__ == "__main__":
    unittest.main()
