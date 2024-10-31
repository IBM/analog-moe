import torch
import datasets
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from sigma_moe import SigmaMoEForCausalLM, SigmaMoEConfiguration
from analog_moe import AnalogSigmaMoELayerAIHWKIT, load_analog_model
from aihwkit.nn.modules.linear import AnalogLinear
from sigma_moe.modeling_sigma_moe import SigmaMoELayer


@torch.no_grad()
def compute_perplexity(model: SigmaMoEForCausalLM, dataloader: DataLoader):
    """
    Compute batched perplexity.
    """
    assert torch.cuda.is_available(), "This function should be run on a GPU" 
    model.cuda()
    model = model.eval()
    total_loss = 0
    total_non_empty = 0
    for inputs in tqdm(dataloader):
        input_ids = inputs["input_ids"]
        input_ids = input_ids.to("cuda")
        labels = inputs["labels"]
        labels = labels[:, 1:].contiguous()
        labels = labels.to("cuda")
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            non_empty_indices = ~(labels == -100).all(1)
            logits = outputs.logits[..., :-1, :].contiguous()
            loss = torch.nn.functional.cross_entropy(
                input=logits[non_empty_indices].transpose(1, 2),
                target=labels[non_empty_indices],
                reduction="none",
            ).sum(1) / (~(labels[non_empty_indices] == -100)).sum(1)
            total_loss += loss.sum()
            total_non_empty += non_empty_indices.sum()
    mean_loss = total_loss / total_non_empty
    return mean_loss.exp()


@torch.no_grad()
def quantize(model):
    n_bits = 5
    for analog_tile in model.analog_tiles():
        w, _ = analog_tile.get_weights()
        abs_max = w.abs().amax(1).view(-1, 1)
        w_norm = w / abs_max
        w_norm = w_norm * (2**(n_bits - 1) - 1)
        w_norm = w_norm.round()
        w_norm = w_norm / (2**(n_bits - 1) - 1)
        w_norm = w_norm * abs_max
        analog_tile.set_weights(w_norm)
    for module in model.modules():
        if isinstance(module, AnalogSigmaMoELayerAIHWKIT):
            keys_abs_max = module.keys.abs().amax(1, keepdim=True)
            values_abs_max = module.values.abs().amax(2, keepdim=True)
            keys = (module.keys / keys_abs_max) * (2**(n_bits - 1) - 1)
            keys = (keys.round() / (2**(n_bits - 1) - 1)) * keys_abs_max
            values = (module.values / values_abs_max) * (2**(n_bits - 1) - 1)
            values = (values.round() / (2**(n_bits - 1) - 1)) * values_abs_max
            module.keys.data = keys
            module.values.data = values
    return model


@torch.no_grad()
def polyval(p, x):
    p = torch.as_tensor(p)
    if isinstance(x, torch.Tensor):
        y = torch.zeros_like(x)
    else:
        x = torch.as_tensor(x)
        y = torch.zeros_like(x)
    for pv in p:
        y = y * x + pv
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    # first, run `huggingface-cli login` and supply a token that has the correct access rights.
    
    # load the dataset. Important: This is already tokenized!
    dataset = datasets.load_dataset("ibm-aimc/sentencepiece-wikitext-103")

    # load the tokenizer (sentencepiece)
    tokenizer = AutoTokenizer.from_pretrained("ibm-aimc/sigma-moe-small")

    # create data loader from it
    dataloader = DataLoader(
        dataset=dataset["test"]["input_ids"],
        batch_size=16,
        shuffle=False,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # we load it from the hub
    model = load_analog_model(
        name="ibm-aimc/analog-sigma-moe-small",
        fp_model_cls=SigmaMoEForCausalLM,
        config_cls=SigmaMoEConfiguration,
        conversion_map={
            torch.nn.Linear: AnalogLinear,
            SigmaMoELayer: AnalogSigmaMoELayerAIHWKIT,
        },
    )

    # model = load_analog_model(
    #     name="ibm-aimc/analog-sigma-dense-small",
    #     fp_model_cls=SigmaMoEForCausalLM,
    #     config_cls=SigmaMoEConfiguration,
    #     conversion_map={
    #         torch.nn.Linear: AnalogLinear,
    #         SigmaMoELayer: AnalogSigmaMoELayerAIHWKIT,
    #     },
    # )

    # model = quantize(model)

    # compute perplexity
    print(f"Perplexity is {compute_perplexity(model, dataloader):.2f}")
