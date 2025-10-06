from transformers import AutoConfig, AutoModelForCausalLM

def build_model(model_name: str, device: str):
    cfg = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(cfg)
    return model.to(device)
