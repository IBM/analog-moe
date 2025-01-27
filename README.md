# Analog MoE

### Julian Büchel, Athanasios Vasilopoulos, William Andrew Simon, Irem Boybat, HsinYu Tsai, Geoffrey W. Burr, Hernan Castro, Bill Filipiak, Manuel Le Gallo, Abbas Rahimi, Vijay Narayanan, Abu Sebastian

_Nature Computational Science, 2025_ [[Article]](https://www.nature.com/articles/s43588-024-00753-x#Sec18)

<div align="center">
  <img src='figures/header.png' width="90%"/>
</div>

Analog MoE is a library that contains GPU kernels for MoE operations extended with hardware-aware training capability. It supports [AIHWKIT-Lightning](https://github.com/IBM/aihwkit-lightning) and [AIHWKIT](https://github.com/IBM/aihwkit). The recommended library to use is AIHWKIT-Lightning. The results from this paper were obtained using AIHWKIT because AIHWKIT-Lightning did not exist yet.

## Requirements
You need to have a GPU which is at least Volta (V100, A100, H100) since this package leverages triton.

## Getting started 🚀
You can create a clean environment using the following
```
conda create -n torch-nightly python=3.10 -y
conda activate torch-nightly
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
conda install -c conda-forge aihwkit-gpu -y
pip install triton transformers datasets
```
Now, you should be able to call `python test_moe_layer.py` and the script should exit without any errors.

## Usage ⚒️
You can convert any aihwkit model and swap out the `SigmaMoELayer`s like so:
```
model = convert_to_analog(
    model,
    rpu_config=<some_rpu_config>,
    conversion_map={
        torch.nn.Linear: AnalogLinear,
        SigmaMoELayer: AnalogSigmaMoELayer
    }
)
```

## Note on `torch.compile`
This layer supports `torch.compile` except when input range learning is enabled since the first `rpu_config.pre_post.input_range.init_from_data`
many samples coming into the layer are used to update the input range in-place which is not supported in torch dynamo.

## Reference 📖
```
@Article{Büchel2025,
  author={B{\"u}chel, Julian
  and Vasilopoulos, Athanasios
  and Simon, William Andrew
  and Boybat, Irem
  and Tsai, HsinYu
  and Burr, Geoffrey W.
  and Castro, Hernan
  and Filipiak, Bill
  and Le Gallo, Manuel
  and Rahimi, Abbas
  and Narayanan, Vijay
  and Sebastian, Abu},
  title={Efficient scaling of large language models with mixture of experts and 3D analog in-memory computing},
  journal={Nature Computational Science},
  year={2025},
  month={Jan},
  day={08},
  issn={2662-8457},
  doi={10.1038/s43588-024-00753-x},
  url={https://doi.org/10.1038/s43588-024-00753-x}
}
```

## License 🔏
Please see the LICENSE file.
