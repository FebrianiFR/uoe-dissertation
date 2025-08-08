import pandas as pd
import pytrec_eval
from doc_util import *
from pyserini import analysis
from gensim.corpora import Dictionary
from gensim.models import LuceneBM25Model
from gensim.similarities import SparseMatrixSimilarity
import json
import os
from query_expansion import *
import openai
from sentence_transformers import SentenceTransformer
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
import pytrec_eval
import torch
import torch.nn.functional as F
from tqdm import tqdm,trange
from openai import OpenAI



# Function to Load Data
def load_data():
    splits = { 'earth_science': 'long_documents/earth_science-00000-of-00001.parquet'}
    df_corpus = pd.read_parquet("hf://datasets/xlangai/BRIGHT/" + splits["earth_science"])
    splits = { 'earth_science': 'examples/earth_science-00000-of-00001.parquet'}
    df_qrels = pd.read_parquet("hf://datasets/xlangai/BRIGHT/" + splits["earth_science"])

    qrels = {}

    # Iterate through each row of the DataFrame
    for index, row in df_qrels[['id', 'gold_ids_long']].iterrows():
        qid = 'q'+row['id']
        # qid = row['id']
        gold_pids = row['gold_ids_long']
        

        # Create the inner dictionary for relevant pids with a label of 1
        # We use a dictionary comprehension to efficiently create this
        relevant_pid_labels = {pid: 1 for pid in gold_pids}

        # Add this to the main qrels dictionary
        qrels[qid] = relevant_pid_labels


    return df_corpus, df_qrels, qrels

# Function to Calculate Retrieval Metrics - from BRIGHT code
def calculate_retrieval_metrics(results, qrels, k_values=[1, 5, 10, 20, 100]):
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    mrr = {"MRR": 0}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

   
    evaluator = pytrec_eval.RelevanceEvaluator(qrels,
                                               {map_string, ndcg_string, recall_string, precision_string, "recip_rank"})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]
        mrr["MRR"] += scores[query_id]["recip_rank"]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)
    mrr["MRR"] = round(mrr["MRR"] / len(scores), 5)

    output = {**ndcg, **_map, **recall, **precision, **mrr}
    print(output)
    return output

# Function to Reformat Score UT before Evaluation Metrics
def reformat_to_qrels_with_iterative_scores(
    input_data: dict,
    start_score: float = 1.0,
    score_decrement_per_item: float = 0.1
) -> dict:
    
    reformatted_qrels = {}

    for qid, pids_list in input_data.items():
        inner_dict = {}
        for i, pid in enumerate(pids_list):
            # Calculate the score for the current PID based on its position (index)
            # Ensure the score doesn't go below zero (or a practical minimum)
            current_score = max(0.0, start_score - (i * score_decrement_per_item))
            inner_dict[pid] = current_score
        reformatted_qrels[qid] = inner_dict
        
    return reformatted_qrels


# BM25 Retriever

def retrieval_bm25(queries, query_ids, documents, doc_ids, excluded_ids, long_context, **kwargs):
    analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())

    
    corpus = []
    for doc in documents:
        if isinstance(doc, str):
            corpus.append(analyzer.analyze(doc))
        else:
            # Handle cases where 'doc' might not be a string (e.g., if it's a list)
            
            try:
                corpus.append(analyzer.analyze(str(doc))) # Attempt to convert to string
                print(f"Warning: Document '{doc}' was converted to string for analysis.")
            except Exception as e:
                raise TypeError(f"Document element expected to be a string, but found type {type(doc)}. Original error: {e}")


    dictionary = Dictionary(corpus)
    model = LuceneBM25Model(dictionary=dictionary, k1=0.9, b=0.4)
    bm25_corpus = model[list(map(dictionary.doc2bow, corpus))]
    bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),
                                        normalize_queries=False, normalize_documents=False)
    all_scores = {}
    bar = tqdm(queries, desc="BM25 retrieval")
    for query_id, query_text in zip(query_ids, queries): 
        bar.update(1)
        
        
        if not isinstance(query_text, str):
            try:
                query_text = str(query_text) # Attempt to convert to string
                print(f"Warning: Query '{query_text}' was converted to string for analysis.")
            except Exception as e:
                raise TypeError(f"Query expected to be a string, but found type {type(query_text)}. Original error: {e}")

        # This call is correct IF 'query_text' is a single string
        analyzed_query = analyzer.analyze(query_text) 
        
        bm25_query = model[dictionary.doc2bow(analyzed_query)]
        similarities = bm25_index[bm25_query].tolist()
        all_scores[str(query_id)] = {}
        for did, s in zip(doc_ids, similarities):
            all_scores[str(query_id)][did] = s
        current_excluded_docs = excluded_ids.get(str(query_id), [])
        
        for did in set(current_excluded_docs): # Iterate over the potentially empty list
            if did != "N/A":
                if did in all_scores[str(query_id)]:
                    all_scores[str(query_id)].pop(did)


        cur_scores = sorted(all_scores[str(query_id)].items(),key=lambda x:x[1],reverse=True)[:100]
        all_scores[str(query_id)] = {}
        for pair in cur_scores:
            all_scores[str(query_id)][pair[0]] = pair[1]
    
    return all_scores



# Sorting JSON embedding files
def sort_json_files_by_number(file_list: list[str]) -> list[str]:
    
    def extract_number(filename):
        # Use regex to find the number before '.json'
        match = re.match(r'(\d+)\.json', filename)
        if match:
            return int(match.group(1))
        
        return float('inf') 

    # Use the extracted number as the key for sorting
    sorted_files = sorted(file_list, key=extract_number)
    return sorted_files

# Function to get score
def get_scores(query_ids,doc_ids,scores,excluded_ids):
    assert len(scores)==len(query_ids),f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0])==len(doc_ids),f"{len(scores[0])}, {len(doc_ids)}"
    emb_scores = {}
    for query_id,doc_scores in zip(query_ids,scores):
        cur_scores = {}
        
        for did,s in zip(doc_ids,doc_scores):
            cur_scores[str(did)] = s
       
        cur_scores = sorted(cur_scores.items(),key=lambda x:x[1],reverse=True)[:100]
        
        emb_scores[str(query_id)] = {}
        for pair in cur_scores:
            emb_scores[str(query_id)][pair[0]] = pair[1]
    return emb_scores

# Access Docuument Embedding - OpenAI
# Embedding get from running the BRIGHT code
def retrieve_doc_openai():
    batch_size = 1024

    document_ids_path = './embeddings/openai/doc_ids/earth_science_False.json'
    path_emb = './embeddings/openai/doc_emb/' 
    embedding_files = os.listdir('./embeddings/openai/doc_emb/')
    embedding_batch_paths =sort_json_files_by_number(embedding_files)
    
    doc_id_to_embedding = {}
    all_doc_ids = []
    all_embeddings = []

    print(f"Loading document IDs from: {document_ids_path}")
    try:
        with open(document_ids_path, 'r') as f:
            all_doc_ids = json.load(f)
        print(f"Loaded {len(all_doc_ids)} document IDs.")
    except FileNotFoundError:
        print(f"Error: Document IDs file not found at {document_ids_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {document_ids_path}")
        return {}

    print("Loading embedding batches...")
    for batch_path in embedding_batch_paths:
        try:
            with open(path_emb+batch_path, 'r') as f:
                batch_embeddings = json.load(f)
                # Ensure each item in the batch is indeed a list (an embedding vector)
                if isinstance(batch_embeddings, list) and all(isinstance(e, list) or isinstance(e, np.ndarray) for e in batch_embeddings):
                    all_embeddings.extend(batch_embeddings)
                else:
                    print(f"Warning: Batch file {batch_path} does not contain a list of embeddings. Skipping.")
        except FileNotFoundError:
            print(f"Error: Embedding batch file not found at {batch_path}")
            continue
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {batch_path}. Skipping.")
            continue
    doc_emb = all_embeddings

    if len(all_doc_ids) == len(doc_emb):
        doc_id_to_embedding = {str(doc_id): emb for doc_id, emb in zip(all_doc_ids, doc_emb)}
        print("Successfully created doc_id_to_embedding map.")
    else:
        print(f"Warning: Mismatch between number of document IDs ({len(all_doc_ids)}) and embeddings ({len(doc_emb)}). Cannot create doc_id_to_embedding map reliably.")


    return doc_emb, doc_id_to_embedding


def retrieval_openai(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,doc_emb,**kwargs):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    new_queries = []
    for q in queries:
        new_queries.append(cut_text_openai(text=q,tokenizer=tokenizer))
    queries = new_queries
    
    batch_size = kwargs.get('batch_size',1024)
    
    document_ids_path = './embeddings/openai/doc_ids/earth_science_False.json'
    path_emb = './embeddings/openai/doc_emb/' # Ensure this path is correct
    embedding_files = os.listdir('./embeddings/openai/doc_emb/')
    embedding_batch_paths =sort_json_files_by_number(embedding_files)
    
    doc_id_to_embedding = {}
    all_doc_ids = []
    all_embeddings = []

    openai_client = OpenAI()


    print(f"Loading document IDs from: {document_ids_path}")
    try:
        with open(document_ids_path, 'r') as f:
            all_doc_ids = json.load(f)
        print(f"Loaded {len(all_doc_ids)} document IDs.")
    except FileNotFoundError:
        print(f"Error: Document IDs file not found at {document_ids_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {document_ids_path}")
        return {}

    doc_emb = doc_emb

    query_emb = []
    for idx in trange(0, len(queries), batch_size):
        cur_emb = get_embedding_openai(texts=queries[idx:idx + batch_size], openai_client=openai_client,
                                       tokenizer=tokenizer)
        query_emb += cur_emb
    scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)



# Function to get document embedding id - for easy access calculating similarity matrix
def get_document_embeddings_by_ids(target_doc_ids, doc_id_to_embedding_map):
    """
    Retrieves document embeddings for a given list of document IDs.

    Args:
        target_doc_ids (list): A list of document IDs (strings or numbers that can be converted to string).
        doc_id_to_embedding_map (dict): The dictionary mapping document IDs to their embeddings.

    Returns:
        dict: A dictionary where keys are the requested document IDs and values are their embeddings.
              If a document ID is not found, it will not be included in the returned dictionary.
    """
    retrieved_embeddings = {}
    for doc_id in target_doc_ids:
        str_doc_id = str(doc_id)
        if str_doc_id in doc_id_to_embedding_map:
            retrieved_embeddings[str_doc_id] = doc_id_to_embedding_map[str_doc_id]
        else:
            print(f"Warning: Document ID '{str_doc_id}' not found in the embedding map.")
    return retrieved_embeddings


# Retriever SBERT
@torch.no_grad()
def retrieval_sbert(queries,query_ids,doc_emb,doc_ids,excluded_ids,**kwargs):
    
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    batch_size = kwargs.get('batch_size',128)
    
    
    
    query_emb = model.encode(queries,show_progress_bar=True,batch_size=batch_size, normalize_embeddings=True)
    scores = cosine_similarity(query_emb, doc_emb)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)

# Retrieve Doc SBERT
def retrived_doc_sbert(all_doc_ids, **kwargs):

    batch_size = kwargs.get('batch_size',128)
    
    cur_cache_file = os.path.join('embeddings/sbert/embeddings-sbert/doc_emb/sbert/earth_science/long_False_128', f'0.npy')
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)

    doc_id_to_embedding = {}

    if len(all_doc_ids) == len(doc_emb):
        doc_id_to_embedding = {str(doc_id): emb for doc_id, emb in zip(all_doc_ids, doc_emb)}
        print("Successfully created doc_id_to_embedding map.")
    else:
        print(f"Warning: Mismatch between number of document IDs ({len(all_doc_ids)}) and embeddings ({len(doc_emb)}). Cannot create doc_id_to_embedding map reliably.")

    
    return doc_emb, doc_id_to_embedding


# Retrieve Doc Google
def retrived_doc_google(all_doc_ids, **kwargs):

    batch_size = kwargs.get('batch_size',8)
    
    cur_cache_file = '/Users/febri/Library/CloudStorage/OneDrive-Personal/Academics/Dissertation/Codes/embeddings/gemini/embeddings-google/doc_emb/google/earth_science/long_False_8.npy'
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)

    doc_id_to_embedding = {}

    if len(all_doc_ids) == len(doc_emb):
        doc_id_to_embedding = {str(doc_id): emb for doc_id, emb in zip(all_doc_ids, doc_emb)}
        print("Successfully created doc_id_to_embedding map.")
    else:
        print(f"Warning: Mismatch between number of document IDs ({len(all_doc_ids)}) and embeddings ({len(doc_emb)}). Cannot create doc_id_to_embedding map reliably.")

    
    return doc_emb, doc_id_to_embedding

# GET EMBEDDING GOOGLE
def get_embedding_google(texts,task,model,dimensionality=768):
    success = False
    while not success:
        try:
            new_texts = []
            for t in texts:
                if t.strip()=='':
                    print('empty content')
                    new_texts.append('empty')
                else:
                    new_texts.append(t)
            texts = new_texts
            inputs = [TextEmbeddingInput(text, task) for text in texts]
            kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
            embeddings = model.get_embeddings(inputs, **kwargs)
            success = True
        except Exception as e:
            print(e)
    return [embedding.values for embedding in embeddings]

# Retriever Google - For Query Embedding
def retrieval_google(queries,query_ids,doc_ids,task,model_id,doc_emb,excluded_ids,**kwargs):
    #os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'balmy-rhino-464718-i5-93469695c135.json'
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    query_emb = []
    
    batch_size = kwargs.get('batch_size',8)
    
        
    for start_idx in tqdm(range(0,len(queries), batch_size),desc='embedding'):
        query_emb += get_embedding_google(texts=queries[start_idx:start_idx+ batch_size],task='RETRIEVAL_QUERY',model=model)
    
    scores = pairwise_cosine_similarity(
    torch.tensor(query_emb, dtype=torch.float32),
    torch.tensor(doc_emb, dtype=torch.float32)
)
    
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)