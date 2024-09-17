from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
import torch
import pandas as pd
from tqdm import tqdm
from typing import List, Optional

def get_embeddings(df: pd.DataFrame, text_columns: List[str], batch_size: int = 32, multilanguage: Optional[bool] = None) -> pd.DataFrame:
    """
    Извлекает эмбеддинги для текстовых данных из заданных столбцов с использованием модели BERT или DistilBERT.

    :param df: Исходный набор данных (pandas DataFrame).
    :param text_columns: Список имен столбцов, содержащих текстовые данные для извлечения эмбеддингов.
    :param batch_size: Размер пакета для обработки текста. По умолчанию 32.
    :param multilanguage: Флаг для использования многоязычной модели BERT. Если True, используется "bert-base-multilingual-cased", иначе используется "distilbert-base-uncased". По умолчанию None.
    :return: Набор данных (pandas DataFrame) с добавленными эмбеддингами и удаленными исходными текстовыми столбцами.
    """
    
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

    return df.drop(text_columns, axis=1)
