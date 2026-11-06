"""Select LoRA parameters and the requested biases for checkpoint saving."""


def collect_lora_state(named_params, bias, gather):
    if bias == "none":
        selected = {name: value for name, value in named_params if "lora_" in name}
    elif bias == "all":
        selected = {name: value for name, value in named_params if "lora_" in name or "bias" in name}
    elif bias == "lora_only":
        selected, biases, associated = {}, {}, set()
        for name, value in named_params:
            if "lora_" in name:
                selected[name] = value
                prefix = name.split("lora_", 1)[0]
                # PEFT wraps the original linear layer as ``base_layer``;
                # older layouts may expose the bias directly on the module.
                associated.update((prefix + "bias", prefix + "base_layer.bias"))
            elif "bias" in name:
                biases[name] = value
        selected.update((name, value) for name, value in biases.items() if name in associated)
    else:
        raise NotImplementedError("unsupported LoRA bias mode: " + str(bias))
    return {name: gather(value, name=name) for name, value in selected.items()}
