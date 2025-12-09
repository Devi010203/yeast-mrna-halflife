## Pretrained RNA-FM model (not included)

The pretrained RNA-FM model is **not** distributed in this data package due to size and licensing considerations.

To run the training and evaluation code, please:

1. Download the pretrained RNA-FM model from its official source  
   (see the original RNA-FM paper / repository for details).

2. Place the downloaded model files into:

   Paper_BioSystems_release/
   └─ data_release/
      └─ model/
         └─ rna-fm/
             config.json
             pytorch_model.bin
             tokenizer.json / vocab files
             ...

3. The training script `code_release/src/train_rnafm_transformer.py` will then load the model via:

   ```python
   RnaFmModel.from_pretrained(config.PRETRAINED_MODEL_NAME, trust_remote_code=True)
   RnaTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME, trust_remote_code=True)