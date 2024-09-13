from transformers import BertTokenizer, BertModel,DistilBertTokenizer, DistilBertModel
import torch
import pandas as pd
from tqdm import tqdm

def get_embeddings(df,text_columns,batch_size=32,multilanguage=None) -> pd.DataFrame:

    if multilanguage:
        model_name = "bert-base-multilingual-cased"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
    else:
        model_name = "distilbert-base-uncased"
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertModel.from_pretrained(model_name)
    
    texts = df[text_columns].fillna('').agg(', '.join, axis=1).tolist()

    embeddings_list = []
    model.eval()  
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)

            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings_list.extend(batch_embeddings.cpu().numpy().tolist())

    col = [f'{i+1}_feature' for i in range(len(embeddings_list[0]))]

    df_new = pd.DataFrame(embeddings_list, columns=col)

    df = pd.concat([df, df_new], axis=1)

    return df.drop(text_columns,axis=1)
