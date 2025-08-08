import numpy as np
import collections
from rank_bm25 import BM25Okapi
import re
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# Function to Normalize Score
def normalize_score(query_scores: dict) -> dict:
    """
    Normalizes a nested dictionary of query-document scores such that
    the scores for each query sum up to 1.

    Args:
        query_scores (dict): A dictionary where keys are query IDs (str)
                             and values are dictionaries of document IDs (str)
                             to their scores (float).

    Returns:
        dict: A new dictionary with the scores normalized.
              If a query has no documents or all scores are zero,
              the scores will remain zero to prevent division by zero.
    """
    normalized_query_scores = {}

    for qid, doc_scores in query_scores.items():
        # Get all scores for the current query
        scores_list = list(doc_scores.values())
        
        # Calculate the sum of scores for this query
        total_sum = sum(scores_list)
        
        normalized_doc_scores = {}
        if total_sum == 0:
            # If sum is zero, keep all scores as zero to avoid division by zero error
            # Or you might choose to exclude this query from the results if appropriate
            for docid, score in doc_scores.items():
                normalized_doc_scores[docid] = 0.0
        else:
            # Normalize each score by dividing by the total sum
            for docid, score in doc_scores.items():
                normalized_doc_scores[docid] = score / total_sum
        
        normalized_query_scores[qid] = normalized_doc_scores
        
    return normalized_query_scores


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalizes a numpy array of scores to sum to 1."""
    if np.sum(scores) == 0:
        return np.zeros_like(scores, dtype=float)
    return scores / np.sum(scores)

# Function Weighted Score Fusion
def weighted_score_fusion(original_scores: dict, expanded_scores: dict, weight_expanded: float = 0.7) -> dict:
    """
    Fuses relevance scores from original and expanded query retrievals.
    A simple weighted average is used here.

    Args:
        original_scores (dict): Dictionary of document names to normalized similarity scores from the original query.
        expanded_scores (dict): Dictionary of document names to normalized similarity scores from the expanded query.
        weight_expanded (float): The weight given to the expanded query scores (0 to 1).
                                 (1 - weight_expanded) will be given to original scores.

    Returns:
        dict: A dictionary of document names to their fused normalized relevance scores.
    """
    results = {}
    for qid, doc_scores in original_scores.items():


        #print(f"\n--- Fusing relevance scores (Expanded weight: {weight_expanded:.2f}) ---")
        fused_scores = {}

        # Get a union of all document names present in either dictionary
        all_doc_names = set(original_scores[qid].keys()).union(set(expanded_scores[qid].keys()))
        # print(len(all_doc_names))

        for doc_name in all_doc_names:
            
            # Get scores, defaulting to 0 if a document is not present in one of the sets
            original_score = original_scores[qid].get(doc_name, 0.0)
            expanded_score = expanded_scores[qid].get(doc_name, 0.0)

            # print(original_score,expanded_score)
            

            fused_scores[doc_name] = (
                (1 - weight_expanded) * original_score +
                weight_expanded * expanded_score
            )
        
        results[qid] = fused_scores
        # print(results)
    return results


# Function Reciprocal Rank Fusion
def reciprocal_rank_score_fusion(
    original_scores: dict,
    expanded_scores: dict,
    k: int = 60,  # A constant to smooth the reciprocal rank. Commonly 60.
    weight_list2: float = 1 # Multiplier for the RRF scores from ranked_list2
) -> dict:
    """
    Performs Reciprocal Rank Fusion (RRF) on two ranked lists of documents (doc_id -> score).
    It also allows applying a weight to the RRF scores derived from the second list.

    Args:
        ranked_list1 (dict): Dictionary of document_id -> score for the first ranked list.
                             Assumes documents are implicitly ordered by score (descending).
        ranked_list2 (dict): Dictionary of document_id -> score for the second ranked list.
                             Assumes documents are implicitly ordered by score (descending).
        k (int): A constant to smooth the reciprocal rank. A common value is 60.
        weight_list2 (float): A multiplier applied to the reciprocal rank scores
                              from `ranked_list2` before summing. Default is 1.0 (no extra weighting).

    Returns:
        dict: A dictionary of document_id -> fused_rrf_score, sorted by score in descending order.
    """
    results = {}
    for qid, doc_scores in original_scores.items():
        fused_scores = collections.defaultdict(float)

        # Process ranked_list1
        # Sort by score in descending order to get ranks
        sorted_list1_items = sorted(original_scores[qid].items(), key=lambda item: item[1], reverse=True)
        for rank, (doc_id, _) in enumerate(sorted_list1_items):
            # Ranks are 0-based, so for RRF formula, add 1.
            
            rrf_score = 1.0 / (k + rank + 1)
            fused_scores[doc_id] += rrf_score

        # Process ranked_list2
        sorted_list2_items = sorted(expanded_scores[qid].items(), key=lambda item: item[1], reverse=True)
        for rank, (doc_id, _) in enumerate(sorted_list2_items):
            rrf_score = 1.0 / (k + rank + 1)
            # Apply the weight for the second list
            fused_scores[doc_id] += rrf_score * weight_list2

        # Sort the final fused scores in descending order
        sorted_fused_scores = dict(sorted(fused_scores.items(), key=lambda item: item[1], reverse=True))
        results[qid] = sorted_fused_scores

    return results

# Function Calculate Diversity Score
def diversity_utility(candidate_doc_idx: int, selected_doc_indices: list, all_doc_sims: np.ndarray) -> float:
    """
    Calculates a diversity score for a candidate document given already selected documents.
    Higher score means more diverse.
    """
    if not selected_doc_indices:
        return 1.0 # Max diversity if no documents have been selected yet

    # Find the maximum similarity between the candidate and any already selected document
    max_sim_to_selected = 0.0
    for selected_idx in selected_doc_indices:
        sim = all_doc_sims[candidate_doc_idx, selected_idx]
        if sim > max_sim_to_selected:
            max_sim_to_selected = sim

    # Diversity is (1 - max_similarity_to_selected). Closer to 1 is more diverse.
    return 1.0 - max_sim_to_selected

# Function to calculate Document Similarity - BM25
def tokenize_text(text):
    """
    Simple tokenizer for text.
    In a real scenario, use a more sophisticated tokenizer (e.g., NLTK, SpaCy).
    """
    # Convert to lowercase, remove punctuation, split by whitespace
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text.split()

def get_bm25_doc_similarity_matrix(documents: dict) -> dict:
    """
    Calculates a document-document similarity matrix using BM25.
    Each document is treated as a query against every other document.

    Args:
        documents (dict): A dictionary where keys are document names (strings)
                          and values are the full text content (strings).

    Returns:
        dict: A nested dictionary where `matrix[doc_name_A][doc_name_B]` is
              the symmetric BM25 similarity score between Document A and Document B,
              normalized between 0 and 1.
    """
    doc_names = list(documents.keys())
    doc_contents = [documents[name] for name in doc_names]

    # Tokenize all documents to create the corpus for BM25
    tokenized_corpus = [tokenize_text(content) for content in doc_contents]

    # Initialize BM25 model with the corpus
    bm25 = BM25Okapi(tokenized_corpus)
    # print("--- BM25 Model Initialized ---")

    # Calculate raw BM25 scores for all pairs
    # This matrix will store BM25_score(doc_i, doc_j)
    raw_bm25_scores = np.zeros((len(doc_names), len(doc_names)))

    for i, doc_name_i in enumerate(doc_names):
        query_tokens = tokenized_corpus[i]
        # Get scores for this doc_i (as query) against ALL other docs (including itself)
        scores_for_query_i = bm25.get_scores(query_tokens)
        raw_bm25_scores[i, :] = scores_for_query_i

        
    # Symmetrize the matrix 
    # A common way is to average S(A,B) and S(B,A)
    symmetric_bm25_scores = (raw_bm25_scores + raw_bm25_scores.T) / 2
    # Ensure self-similarity is 1.0 
    np.fill_diagonal(symmetric_bm25_scores, np.max(symmetric_bm25_scores)) 

    # Normalize the symmetric scores to 0-1 range
    
    min_score = np.min(symmetric_bm25_scores)
    max_score = np.max(symmetric_bm25_scores)
    if max_score == min_score: # Avoid division by zero
        normalized_bm25_scores = np.ones_like(symmetric_bm25_scores) * 0.5 
    else:
        normalized_bm25_scores = (symmetric_bm25_scores - min_score) / (max_score - min_score)
    
    # Re-map to dictionary format
    bm25_doc_similarity_matrix = {name_i: {} for name_i in doc_names}
    for i, doc_name_i in enumerate(doc_names):
        for j, doc_name_j in enumerate(doc_names):
            bm25_doc_similarity_matrix[doc_name_i][doc_name_j] = normalized_bm25_scores[i, j]

    
    return bm25_doc_similarity_matrix




# Function Document Utility
def document_utility_selection_bm25(
    fused_score: np.ndarray, 
    all_doc_sims: np.ndarray,             
    doc_names: list,                      
    k: int = 10,                           
    alpha: float = 0.7                    # Weight for relevance (0 to 1). Higher alpha = more emphasis on relevance.
) -> list:
    """
    Performs document selection based on Bayesian information utility gain.
    It iteratively selects documents that maximize Expected Utility, balancing
    relevance (from fused priors) and diversity.

    Args:
        fused_prior_probabilities (np.ndarray): A 1D numpy array of fused and normalized
                                                 relevance probabilities for each document.
                                                 Indices should align with `doc_names`.
        all_doc_sims (np.ndarray): A 2D numpy array (matrix) of normalized
                                   document-document similarities.
        doc_names (list): A list of strings, where each string is the name of a document.
                          The order of names must correspond to the indices in
                          `fused_prior_probabilities` and `all_doc_sims`.
        k (int): The number of top documents to retrieve.
        alpha (float): The weight given to relevance in the utility function.
                       (1 - alpha) will be given to diversity.

    Returns:
        list: A list of `k` selected document names.
    """
    if not isinstance(fused_score, np.ndarray) or fused_score.ndim != 1:
        raise ValueError("fused_prior_probabilities must be a 1D numpy array.")
    if not isinstance(all_doc_sims, np.ndarray) or all_doc_sims.ndim != 2:
        raise ValueError("all_doc_sims must be a 2D numpy array (matrix).")
    if fused_score.shape[0] != all_doc_sims.shape[0] or all_doc_sims.shape[0] != all_doc_sims.shape[1]:
        raise ValueError("Shape mismatch: fused_prior_probabilities length must match all_doc_sims dimensions.")

    num_docs = len(doc_names)
    selected_documents_names = []  # Stores names of selected documents
    selected_doc_indices = []      # Stores integer indices of selected documents for diversity calculation
    candidate_indices = list(range(num_docs)) # Start with all docs as candidates

    results = {}
    for step in range(k):
        if not candidate_indices:
            # print("No more candidates to select from.")
            break

        best_expected_utility = -1.0
        best_candidate_idx = -1

        
        for current_candidate_idx in candidate_indices:
            candidate_doc_name = doc_names[current_candidate_idx]

            
            P_doc_relevant = fused_score[current_candidate_idx]

            
            relevance_utility = P_doc_relevant

            # U_Diversity: How unique is it compared to what's already selected?
            du = diversity_utility(
                current_candidate_idx,
                selected_doc_indices, # List of indices of already chosen docs
                all_doc_sims
            )

            # Combined Utility: Weighted sum of relevance and diversity
            # Higher alpha favors relevance, lower alpha favors diversity
            combined_utility= (alpha * relevance_utility) + ((1 - alpha) * du)

            # Expected Utility (EU): P(Relevant) * U(Combined)
            expected_utility = P_doc_relevant * combined_utility

            if expected_utility > best_expected_utility:
                best_expected_utility = expected_utility
                best_candidate_idx = current_candidate_idx
            elif expected_utility == best_expected_utility:
                # Tie-breaking: if EUs are equal, prefer the one with higher raw relevance
                # or a stable tie-breaking like lower index to ensure determinism
                if fused_score[current_candidate_idx] > fused_score[best_candidate_idx]:
                    best_candidate_idx = current_candidate_idx


        if best_candidate_idx != -1:
            selected_documents_names.append(doc_names[best_candidate_idx])
            selected_doc_indices.append(best_candidate_idx)
            candidate_indices.remove(best_candidate_idx)
            results[doc_names[best_candidate_idx]] = best_expected_utility
            
        else:
            pass
            
    return results, selected_documents_names

def reformat_to_qrels_with_iterative_scores(
    input_data: dict,
    start_score: float = 1.0,
    score_decrement_per_item: float = 0.1
) -> dict:
    """
    Reformats a dictionary from {qid: [pid1, pid2, ...]}
    to {qid: {pid1: score1, pid2: score2, ...}},
    where scores iteratively decrease for subsequent items in the list.

    Args:
        input_data (dict): The input dictionary with query IDs as keys and
                           lists of document IDs as values.
        start_score (float): The score to assign to the first document in each list.
        score_decrement_per_item (float): The amount by which the score decreases
                                          for each subsequent document in the list.

    Returns:
        dict: The reformatted dictionary in the desired qrels format.
    """
    reformatted_qrels = {}

    for qid, pids_list in input_data.items():
        inner_dict = {}
        for i, pid in enumerate(pids_list):
            
            current_score = max(0.0, start_score - (i * score_decrement_per_item))
            inner_dict[pid] = current_score
        reformatted_qrels[qid] = inner_dict
        
    return reformatted_qrels

def get_candidate_pairwise_similarity_dict(candidate_doc_ids, doc_id_to_embedding_map):
    """
    Computes and returns the pairwise cosine similarity matrix for a given set of candidate documents
    in a nested dictionary format. The matrix is symmetrized and normalized to 0-1.

    Args:
        candidate_doc_ids (list): A list of document IDs (strings) that are candidates for pairwise comparison.
                                  These IDs are typically the top-N retrieved documents for a query.
        doc_id_to_embedding_map (dict): The pre-loaded dictionary mapping ALL document IDs to their embeddings.

    Returns:
        dict: A nested dictionary where `doc_similarity_matrix[doc_name_i][doc_name_j]` provides
              the normalized similarity score between `doc_name_i` and `doc_name_j`.
              Returns an empty dict if no valid candidates.
    """
    candidate_embeddings_list = []
    
    actual_candidate_ids_in_order = [] 

    for doc_id in candidate_doc_ids:
        str_doc_id = str(doc_id)
        if str_doc_id in doc_id_to_embedding_map:
            candidate_embeddings_list.append(doc_id_to_embedding_map[str_doc_id])
            actual_candidate_ids_in_order.append(str_doc_id)
        else:
            print(f"Warning: Candidate Document ID '{str_doc_id}' not found in the embedding map. Skipping for pairwise comparison.")
    
    if not candidate_embeddings_list:
        print("No valid candidate embeddings found to compute pairwise similarity.")
        return {}

    # Convert the list of selected candidate embeddings to a NumPy array
    candidate_embeddings_array = np.array(candidate_embeddings_list)

    # Compute raw pairwise cosine similarity ONLY for these candidates
    raw_cosine_scores = cosine_similarity(candidate_embeddings_array)

    # Symmetrize the matrix (optional but good practice for "similarity")
    
    symmetric_cosine_scores = (raw_cosine_scores + raw_cosine_scores.T) / 2
    
    
    np.fill_diagonal(symmetric_cosine_scores, 1.0)

    # Normalize the symmetric scores to 0-1 range
    
    min_score = np.min(symmetric_cosine_scores)
    max_score = np.max(symmetric_cosine_scores)
    
    if max_score == min_score: # Avoid division by zero
        normalized_cosine_scores = np.ones_like(symmetric_cosine_scores) * 0.5 # Neutral if all same
    else:
        normalized_cosine_scores = (symmetric_cosine_scores - min_score) / (max_score - min_score)
    
    # Re-map to dictionary format
    doc_similarity_matrix_dict = {name_i: {} for name_i in actual_candidate_ids_in_order}
    for i, doc_name_i in enumerate(actual_candidate_ids_in_order):
        for j, doc_name_j in enumerate(actual_candidate_ids_in_order):
            doc_similarity_matrix_dict[doc_name_i][doc_name_j] = normalized_cosine_scores[i, j]

    return doc_similarity_matrix_dict


def save_result(type, model, workflow, file_to_save):
    with open(f'./{type}/{model}_{workflow}.json', 'w') as d:
        json.dump(file_to_save, d)


# Function Document Utility using embedding
def document_utility_dense(
    fused_prior_probabilities: np.ndarray, 
    all_doc_sims: np.ndarray,             
    doc_names: list,                      
    k: int = 10,                           
    alpha: float = 0.7                    
) -> list:
    """
    Performs document selection based on Bayesian information utility gain.
    It iteratively selects documents that maximize Expected Utility, balancing
    relevance (from fused priors) and diversity.

    Args:
        fused_prior_probabilities (np.ndarray): A 1D numpy array of fused and normalized
                                                 relevance probabilities for each document.
                                                 Indices should align with `doc_names`.
        all_doc_sims (np.ndarray): A 2D numpy array (matrix) of normalized
                                   document-document similarities.
        doc_names (list): A list of strings, where each string is the name of a document.
                          The order of names must correspond to the indices in
                          `fused_prior_probabilities` and `all_doc_sims`.
        k (int): The number of top documents to retrieve.
        alpha (float): The weight given to relevance in the utility function.
                       (1 - alpha) will be given to diversity.

    Returns:
        list: A list of `k` selected document names.
    """
    if not isinstance(fused_prior_probabilities, np.ndarray) or fused_prior_probabilities.ndim != 1:
        raise ValueError("fused_prior_probabilities must be a 1D numpy array.")
    if not isinstance(all_doc_sims, np.ndarray) or all_doc_sims.ndim != 2:
        raise ValueError("all_doc_sims must be a 2D numpy array (matrix).")
    if fused_prior_probabilities.shape[0] != all_doc_sims.shape[0] or all_doc_sims.shape[0] != all_doc_sims.shape[1]:
        raise ValueError("Shape mismatch: fused_prior_probabilities length must match all_doc_sims dimensions.")

    num_docs = len(doc_names)
    selected_documents_names = []  # Stores names of selected documents
    selected_doc_indices = []      # Stores integer indices of selected documents for diversity calculation
    candidate_indices = list(range(num_docs)) # Start with all docs as candidates

    results = {}

    for step in range(k):
        if not candidate_indices:
            print("No more candidates to select from.")
            break

        best_expected_utility = -1.0
        best_candidate_idx = -1

        for current_candidate_idx in candidate_indices:
            candidate_doc_name = doc_names[current_candidate_idx]

            max_fused_prob_in_candidates = 0.0
            for idx in candidate_indices: # iterate only over remaining candidates
                max_fused_prob_in_candidates = max(max_fused_prob_in_candidates, fused_prior_probabilities[idx])

            if max_fused_prob_in_candidates > 0:
                # Scale U_relevance for the current candidate
                U_relevance_current_candidate = fused_prior_probabilities[current_candidate_idx] / max_fused_prob_in_candidates
            else:
                U_relevance_current_candidate = 0.0 # Or some small epsilon if all are zero

            relevance_utility = U_relevance_current_candidate

            # U_Diversity: How unique is it compared to what's already selected?
            du = diversity_utility(
                current_candidate_idx,
                selected_doc_indices, # List of indices of already chosen docs
                all_doc_sims
            )

            # Combined Utility: Weighted sum of relevance and diversity
            # Higher alpha favors relevance, lower alpha favors diversity
            combined_utility = (alpha * relevance_utility) + ((1 - alpha) * du)

            # Document Utility (EU): P(Relevant) * U(Combined)
            expected_utility = relevance_utility * combined_utility

            if expected_utility > best_expected_utility:
                best_expected_utility = expected_utility
                best_candidate_idx = current_candidate_idx
            elif expected_utility == best_expected_utility:
                # Tie-breaking: if EUs are equal, prefer the one with higher raw relevance
                # or a stable tie-breaking like lower index to ensure determinism
                if fused_prior_probabilities[current_candidate_idx] > fused_prior_probabilities[best_candidate_idx]:
                    best_candidate_idx = current_candidate_idx
            # time.sleep(1)


        if best_candidate_idx != -1:
            selected_documents_names.append(doc_names[best_candidate_idx])
            selected_doc_indices.append(best_candidate_idx)
            candidate_indices.remove(best_candidate_idx)
            results[doc_names[best_candidate_idx]] = best_expected_utility
               

    return results, selected_documents_names



def maximal_marginal_relevance(query_id, sim_scores, doc_df, lambda_param, top_k, scale_to_100=True, verbose=False):
    doc_scores = sim_scores[query_id]
    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Filter content for available docs
    content_map = dict(zip(doc_df['id'], doc_df['content']))
    selected_docs = [doc_id for doc_id, _ in ranked_docs if doc_id in content_map]
    docs_content = [content_map[doc_id] for doc_id in selected_docs]
    relevance_scores = {doc_id: score for doc_id, score in ranked_docs if doc_id in content_map}

    if len(selected_docs) == 0:
        return {}

    # Important: now doc_ids matches the TF-IDF matrix
    doc_ids = selected_docs
    tfidf = TfidfVectorizer().fit_transform(docs_content)
    doc_sim_matrix = cosine_similarity(tfidf)

    selected = []
    remaining = doc_ids.copy()
    mmr_score_map = {}

    while len(selected) < min(top_k, len(remaining)):
        mmr_scores = []
        for doc_id in remaining:
            rel = relevance_scores[doc_id]
            if not selected:
                diversity = 0
            else:
                candidate_index = doc_ids.index(doc_id)
                selected_indices = [doc_ids.index(sel) for sel in selected]
                diversity = max(doc_sim_matrix[candidate_index][j] for j in selected_indices)
            mmr = lambda_param * rel - (1 - lambda_param) * diversity
            mmr_scores.append((doc_id, mmr))

            # if verbose:
            #     print(f"[MMR Calc] Doc: {doc_id}, Rel: {rel:.4f}, Div: {diversity:.4f}, MMR: {mmr:.4f}")

        next_doc, next_score = max(mmr_scores, key=lambda x: x[1])
        selected.append(next_doc)
        mmr_score_map[next_doc] = next_score
        remaining.remove(next_doc)

    return mmr_score_map