from transformers import AutoTokenizer, RobertaModel

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-10M-MTR")

model = RobertaModel.from_pretrained("DeepChem/ChemBERTa-10M-MTR")

'a'