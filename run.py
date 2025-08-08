#import library
from retrievers import *
from query_expansion import *
import openai
import json

# Acess Environment Config
with open('config.json', 'r') as f:
    configs = json.load(f)

openai.api_key = configs['OPENAI_API_KEY']
os.environ['OPENAI_API_KEY'] = configs['OPENAI_API_KEY']



# # # Load Data
df_corpus, df_qrels, qrels = load_data()



# Query Expansion
df_queries = query_expansion(df_queries=df_qrels,LLM_model='gpt-4.1-2025-04-14')
df_queries.to_csv('query_expansion_result.csv', index=False)



#################################################################
# BM25 Retrieval - Baseline
method = {'original':'query',
          'general' :'expanded_general',
          'domain'  :'expanded_focus'}




# Define Lists
documents = df_corpus['content'].to_list()
doc_ids = df_corpus['id'].to_list()
df_queries['id'] = 'q'+df_queries['id'].astype(str)
query_ids = df_queries['id'].to_list()
sample_excluded_ids = {}


# BM25 Evaluation
# #################################################################
print('BM25')
score_bm25 = {}
for k,v in method.items():
    queries = df_queries[v].to_list()

    # BM25 Run
    bm25_result = retrieval_bm25(
            queries=queries,
            query_ids=query_ids,
            documents=documents,
            doc_ids=doc_ids,
            excluded_ids=sample_excluded_ids,
            long_context=None
        )

    # Save calculated scores from BM25
    with open(f'./scores/bm25_{v}.json', 'w') as d:
        json.dump(bm25_result, d)

    # Normalized Score
    score_bm25[k] = normalize_score(bm25_result)

    #BM25 Run Result Calculated Metrics
    print('Metrics of:',v)
    bm25_metrics = calculate_retrieval_metrics(bm25_result, qrels)

    with open(f'./results/bm25_{v}.json', 'w') as d:
        json.dump(bm25_metrics, d)
    
#################################################################


# Fusion Score with Two Distinct Methods
# Weighted Score Fusion - General
print(len(score_bm25['original']), len(score_bm25['general']))
wsf_ori_gen = weighted_score_fusion(score_bm25['original'], score_bm25['general'])
save_result('scores','bm25','wsf_general', wsf_ori_gen)


print('Metrics of WSF Original & General Expansion')
wsf_ori_gen_metrics = calculate_retrieval_metrics(wsf_ori_gen, qrels)
save_result('results','bm25','wsf_general', wsf_ori_gen_metrics)


#Weighted Score Fusion - Domain Focused
wsf_ori_dom = weighted_score_fusion(score_bm25['original'], score_bm25['domain'])
save_result('scores','bm25','wsf_domain', wsf_ori_dom)
print('Metrics of WSF Original & Domain Expansion')
wsf_ori_dom_metrics = calculate_retrieval_metrics(wsf_ori_dom, qrels)
save_result('results','bm25','wsf_domain', wsf_ori_dom_metrics)


# #################################################################
# Reciprocal Rank Fusion - General
print(len(score_bm25['original']), len(score_bm25['general']))
rrf_ori_gen = reciprocal_rank_score_fusion(score_bm25['original'], score_bm25['general'])
save_result('scores','bm25','rrf_general', rrf_ori_gen)


print('Metrics of RRF Original & General Expansion')
rrf_ori_gen_metrics = calculate_retrieval_metrics(rrf_ori_gen, qrels)
save_result('results','bm25','rrf_general', rrf_ori_gen_metrics)


# Reciprocal Rank Score Fusion - Domain Focused
rrf_ori_dom = reciprocal_rank_score_fusion(score_bm25['original'], score_bm25['domain'])
save_result('scores','bm25','rrf_domain', rrf_ori_dom)
print('Metrics of RRF Original & Domain Expansion')
rrf_ori_dom_metrics = calculate_retrieval_metrics(rrf_ori_dom, qrels)
save_result('results','bm25','rrf_domain', rrf_ori_dom_metrics)



# Save Result as Dictionary
score_fused = {'wsf_gen':wsf_ori_gen,
               'wsf_dom':wsf_ori_dom,
               'rrf_gen':rrf_ori_gen,
               'rrf_dom':rrf_ori_dom}

# #################################################################
# Document Utility Workflow

for k,v in score_fused.items():
    results_per_query = {}
    du_res_q = {}
    score = v
    for query_id in tqdm(query_ids):
        
        top_m_candidate_names = [name for name in score[query_id].keys()]
        M = len(top_m_candidate_names)
        
        # Get the top document 
        top_m_candidate_contents = {name: df_corpus[df_corpus['id']==name]['content'].iloc[0] for name in top_m_candidate_names}


        # Generate the M x M similarity matrix for ONLY these M candidates
        
        bm25_sim_matrix_M_by_M = get_bm25_doc_similarity_matrix(top_m_candidate_contents)

       
        num_m_docs = M
        all_doc_sims_M_by_M_array = np.zeros((num_m_docs, num_m_docs))
        for i, doc_name_i in enumerate(top_m_candidate_names):
            for j, doc_name_j in enumerate(top_m_candidate_names):
                # Ensure bm25_sim_matrix_M_by_M has all pairs for these M docs
                all_doc_sims_M_by_M_array[i, j] = bm25_sim_matrix_M_by_M[doc_name_i][doc_name_j]

        # Get the fused scores for the current query
        current_query_fused_scores = score.get(query_id, {})
        if not current_query_fused_scores:
            print(f"No fused scores found for query ID: {query_id}. Skipping.")
            results_per_query[query_id] = []
            continue

        # Convert current query's fused scores to a NumPy array, aligning with all_unique_doc_names_in_corpus order
        fused_prior_probabilities_array = np.array([
            current_query_fused_scores.get(doc_name, 0.0) # Default to 0.0 if doc not in this query's top fusion
            for doc_name in top_m_candidate_names
        ])

        # Perform Bayesian Information Utility - caclulate document utility
        du_res, selected_docs_for_query = document_utility_selection_bm25(
            fused_score=fused_prior_probabilities_array,
            all_doc_sims=all_doc_sims_M_by_M_array,
            doc_names=list(top_m_candidate_names),
            k=10,         
            alpha=0.7    # Adjust alpha for each query if needed, or keep constant
        )
        du_res_q[query_id]=du_res
        results_per_query[query_id] = selected_docs_for_query

    save_result('scores','bm25_docutil',k,du_res_q)

    print('Document Utility Result:',k)
    metric = calculate_retrieval_metrics(du_res_q,qrels,k_values=[1,5,10])
    save_result('results','bm25_docutil',k,metric)


# # #################################################################
# OPEN AI MODEL
print('OPENAI')
score_oai = {}
doc_emb, doc_id_to_emb = retrieve_doc_openai()

# Baseline
for k,v in method.items():
    queries = df_queries[v].to_list()
    score = retrieval_openai(queries=queries, query_ids=query_ids, doc_ids=doc_ids, excluded_ids={},cache_dir='./embeddings/openai/doc_emb', long_context=False,doc_emb=doc_emb, documents = documents, task='earth_science', model_id='openai')
    score_oai[k] = normalize_score(score)
    results = calculate_retrieval_metrics(score, qrels, k_values=[1,5,10,20,25,100])
    save_result(type='scores',model='openai',workflow=v,file_to_save=score)
    save_result(type='results',model='openai',workflow=v,file_to_save=results)

# WSF
# WSF - Original
wsf_ori_gen_oai = weighted_score_fusion(score_oai['original'], score_oai['general'])
save_result(type='scores',model='openai',workflow='wsf_general',file_to_save=wsf_ori_gen_oai)
wsf_ori_gen_oai_metrics = calculate_retrieval_metrics(wsf_ori_gen_oai, qrels)
save_result(type='results',model='openai',workflow='wsf_general',file_to_save=wsf_ori_gen_oai_metrics)

# WSF - Domain
wsf_ori_dom_oai = weighted_score_fusion(score_oai['original'], score_oai['domain'])
save_result(type='scores',model='openai',workflow='wsf_domain',file_to_save=wsf_ori_dom_oai)
wsf_ori_dom_oai_metrics = calculate_retrieval_metrics(wsf_ori_dom_oai, qrels)
save_result(type='results',model='openai',workflow='wsf_domain',file_to_save=wsf_ori_dom_oai_metrics)

# RRF
# RRF - Original
rrf_ori_gen_oai = reciprocal_rank_score_fusion(score_oai['original'], score_oai['general'])
save_result(type='scores',model='openai',workflow='rrf_general',file_to_save=rrf_ori_gen_oai)
rrf_ori_gen_oai_metrics = calculate_retrieval_metrics(rrf_ori_gen_oai, qrels)
save_result(type='results',model='openai',workflow='rrf_general',file_to_save=rrf_ori_gen_oai_metrics)

# RRF - Domain
rrf_ori_dom_oai = reciprocal_rank_score_fusion(score_oai['original'], score_oai['domain'])
save_result(type='scores',model='openai',workflow='rrf_domain',file_to_save=rrf_ori_dom_oai)
rrf_ori_dom_oai_metrics = calculate_retrieval_metrics(rrf_ori_dom_oai, qrels)
save_result(type='results',model='openai',workflow='rrf_domain',file_to_save=rrf_ori_dom_oai_metrics)


## Document Utility
# Accessing document Embedding

openai_doc = {'wsf_gen':wsf_ori_gen_oai, 
              'wsf_dom':wsf_ori_dom_oai, 
              'rrf_gen':rrf_ori_gen_oai, 
              'rrf_dom':rrf_ori_dom_oai}

# Iterate through the combination
for k,v in openai_doc.items():
    fuse_score_dict = v
    results_per_query = {}
    du_res_q_oai = {}
    print(k)
    for query_id in query_ids:
    
        top_m_candidate_names = [name for name in fuse_score_dict[query_id].keys()]
        M = len(top_m_candidate_names)
        top_m_candidate_contents = {name: df_corpus[df_corpus['id']==name]['content'].iloc[0] for name in top_m_candidate_names}


        # Generate the M x M similarity matrix for ONLY these M candidates
      
        sim_matrix_M_by_M = get_candidate_pairwise_similarity_dict(top_m_candidate_contents,doc_id_to_emb)
        
        #    Use top_m_candidates_names as the consistent order
        num_m_docs = M
        all_doc_sims_M_by_M_array = np.zeros((num_m_docs, num_m_docs))
        for i, doc_name_i in enumerate(top_m_candidate_names):
            for j, doc_name_j in enumerate(top_m_candidate_names):
                # Ensure sim_matrix_M_by_M has all pairs for these M docs
                all_doc_sims_M_by_M_array[i, j] = sim_matrix_M_by_M[doc_name_i][doc_name_j]

        # Get the fused scores for the current query
        current_query_fused_scores = fuse_score_dict.get(query_id, {})
        if not current_query_fused_scores:
            print(f"No fused scores found for query ID: {query_id}. Skipping.")
            results_per_query[query_id] = []
            continue

        # Convert current query's fused scores to a NumPy array, aligning with all_unique_doc_names_in_corpus order
        fused_prior_probabilities_array = np.array([
            current_query_fused_scores.get(doc_name, 0.0) # Default to 0.0 if doc not in this query's top fusion
            for doc_name in top_m_candidate_names
        ])

        fused_prior_probabilities_array = normalize_scores(fused_prior_probabilities_array)
        
        # Calculate Document Utility
        res, selected_docs_for_query = document_utility_dense(
            fused_prior_probabilities=fused_prior_probabilities_array,
            all_doc_sims=all_doc_sims_M_by_M_array,
            doc_names=list(top_m_candidate_names),
            k=10,         # top 10 documents
            alpha=0.7    # Adjust alpha for each query if needed, or keep constant
        )

        results_per_query[query_id] = selected_docs_for_query
        du_res_q_oai[query_id]=res

    temp_score = calculate_retrieval_metrics(du_res_q_oai,qrels,k_values=[1,5,10])
    save_result('scores','openai','docutil_'+k,du_res_q_oai)
    save_result('results','openai','docutil_'+k,temp_score)

###################################################################################
# SBERT MODEL
print('SBERT')

score_sbert = {}
doc_emb, doc_id_to_emb = retrived_doc_sbert(doc_ids)
for k,v in method.items():
    queries = df_queries[v].to_list()
    score = retrieval_sbert(queries=queries, query_ids=query_ids, doc_ids=doc_ids, excluded_ids={}, doc_emb = doc_emb, task='earth_science', model_id='sbert')
    score_sbert[k] = normalize_score(score)
    results = calculate_retrieval_metrics(score, qrels, k_values=[1,5,10,20,25,100])
    save_result(type='scores',model='sbert',workflow=v,file_to_save=score)
    save_result(type='results',model='sbert',workflow=v,file_to_save=results)

# WSF
# WSF - Original
wsf_ori_gen_sbert = weighted_score_fusion(score_sbert['original'], score_sbert['general'])
save_result(type='scores',model='sbert',workflow='wsf_general',file_to_save=wsf_ori_gen_sbert)
wsf_ori_gen_sbert_metrics = calculate_retrieval_metrics(wsf_ori_gen_sbert, qrels)
save_result(type='results',model='sbert',workflow='wsf_general',file_to_save=wsf_ori_gen_sbert_metrics)

# WSF - Domain
wsf_ori_dom_sbert = weighted_score_fusion(score_sbert['original'], score_sbert['domain'])
save_result(type='scores',model='sbert',workflow='wsf_domain',file_to_save=wsf_ori_dom_sbert)
wsf_ori_dom_sbert_metrics = calculate_retrieval_metrics(wsf_ori_dom_sbert, qrels)
save_result(type='results',model='sbert',workflow='wsf_domain',file_to_save=wsf_ori_dom_sbert_metrics)

# RRF
# RRF - Original
rrf_ori_gen_sbert = reciprocal_rank_score_fusion(score_sbert['original'], score_sbert['general'])
save_result(type='scores',model='sbert',workflow='rrf_general',file_to_save=rrf_ori_gen_sbert)
rrf_ori_gen_sbert_metrics = calculate_retrieval_metrics(rrf_ori_gen_sbert, qrels)
save_result(type='results',model='sbert',workflow='rrf_general',file_to_save=rrf_ori_gen_sbert_metrics)

# RRF - Domain
rrf_ori_dom_sbert = reciprocal_rank_score_fusion(score_sbert['original'], score_sbert['domain'])
save_result(type='scores',model='sbert',workflow='rrf_domain',file_to_save=rrf_ori_dom_sbert)
rrf_ori_dom_sbert_metrics = calculate_retrieval_metrics(rrf_ori_dom_sbert, qrels)
save_result(type='results',model='sbert',workflow='rrf_domain',file_to_save=rrf_ori_dom_sbert_metrics)


# Document Utility Workflow
sbert_doc = {'wsf_gen':wsf_ori_gen_sbert, 
              'wsf_dom':wsf_ori_dom_sbert, 
              'rrf_gen':rrf_ori_gen_sbert, 
              'rrf_dom':rrf_ori_dom_sbert}

# Iterate through the combination
for k,v in sbert_doc.items():
    fuse_score_dict = v
    results_per_query = {}
    du_res_q_oai = {}
    print(k)
    for query_id in query_ids:
    
        top_m_candidate_names = [name for name in fuse_score_dict[query_id].keys()]
        M = len(top_m_candidate_names)
        top_m_candidate_contents = {name: df_corpus[df_corpus['id']==name]['content'].iloc[0] for name in top_m_candidate_names}


        # Generate the M x M similarity matrix for ONLY these M candidates
      
        sim_matrix_M_by_M = get_candidate_pairwise_similarity_dict(top_m_candidate_contents,doc_id_to_emb)
        
        # Use top_m_candidates_names as the consistent order
        num_m_docs = M
        all_doc_sims_M_by_M_array = np.zeros((num_m_docs, num_m_docs))
        for i, doc_name_i in enumerate(top_m_candidate_names):
            for j, doc_name_j in enumerate(top_m_candidate_names):
                # Ensure sim_matrix_M_by_M has all pairs for these M docs
                all_doc_sims_M_by_M_array[i, j] = sim_matrix_M_by_M[doc_name_i][doc_name_j]

        # Get the fused scores for the current query
        current_query_fused_scores = fuse_score_dict.get(query_id, {})
        if not current_query_fused_scores:
            print(f"No fused scores found for query ID: {query_id}. Skipping.")
            results_per_query[query_id] = []
            continue

        # Convert current query's fused scores to a NumPy array, aligning with all_unique_doc_names_in_corpus order
        fused_prior_probabilities_array = np.array([
            current_query_fused_scores.get(doc_name, 0.0) # Default to 0.0 if doc not in this query's top fusion
            for doc_name in top_m_candidate_names
        ])

        fused_prior_probabilities_array = normalize_scores(fused_prior_probabilities_array)
        
        # 3. Perform Document Utility
        res, selected_docs_for_query = document_utility_dense(
            fused_prior_probabilities=fused_prior_probabilities_array,
            all_doc_sims=all_doc_sims_M_by_M_array,
            doc_names=list(top_m_candidate_names),
            k=10,         # Get the top 10 documents
            alpha=0.7    # Adjust alpha for each query if needed, or keep constant
        )

        results_per_query[query_id] = selected_docs_for_query
        du_res_q_oai[query_id]=res

    temp_score = calculate_retrieval_metrics(du_res_q_oai,qrels,k_values=[1,5,10])
    save_result('scores','sbert','docutil_'+k,du_res_q_oai)
    save_result('results','sbert','docutil_'+k,temp_score)

####################################################################################
# Gemini Model
print('GOOGLE')

score_google = {}
doc_emb, doc_id_to_emb = retrived_doc_google(doc_ids)
for k,v in method.items():
    queries = df_queries[v].to_list()
    score = retrieval_google(queries=queries, query_ids=query_ids, doc_ids=doc_ids, excluded_ids={}, doc_emb = doc_emb, task='earth_science', model_id='google')
    score_google[k] = normalize_score(score)
    results = calculate_retrieval_metrics(score, qrels, k_values=[1,5,10,20,25,100])
    save_result(type='scores',model='google',workflow=v,file_to_save=score)
    save_result(type='results',model='google',workflow=v,file_to_save=results)

# WSF
# WSF - Original
wsf_ori_gen_google = weighted_score_fusion(score_google['original'], score_google['general'])
save_result(type='scores',model='google',workflow='wsf_general',file_to_save=wsf_ori_gen_google)
wsf_ori_gen_google_metrics = calculate_retrieval_metrics(wsf_ori_gen_google, qrels)
save_result(type='results',model='google',workflow='wsf_general',file_to_save=wsf_ori_gen_google_metrics)

# WSF - Domain
wsf_ori_dom_google = weighted_score_fusion(score_google['original'], score_google['domain'])
save_result(type='scores',model='google',workflow='wsf_domain',file_to_save=wsf_ori_dom_google)
wsf_ori_dom_google_metrics = calculate_retrieval_metrics(wsf_ori_dom_google, qrels)
save_result(type='results',model='google',workflow='wsf_domain',file_to_save=wsf_ori_dom_google_metrics)

# RRF
# RRF - Original
rrf_ori_gen_google = reciprocal_rank_score_fusion(score_google['original'], score_google['general'])
save_result(type='scores',model='google',workflow='rrf_general',file_to_save=rrf_ori_gen_google)
rrf_ori_gen_google_metrics = calculate_retrieval_metrics(rrf_ori_gen_google, qrels)
save_result(type='results',model='google',workflow='rrf_general',file_to_save=rrf_ori_gen_google_metrics)

# RRF - Domain
rrf_ori_dom_google = reciprocal_rank_score_fusion(score_google['original'], score_google['domain'])
save_result(type='scores',model='google',workflow='rrf_domain',file_to_save=rrf_ori_dom_google)
rrf_ori_dom_google_metrics = calculate_retrieval_metrics(rrf_ori_dom_google, qrels)
save_result(type='results',model='google',workflow='rrf_domain',file_to_save=rrf_ori_dom_google_metrics)


# # Document Utility Workflow
google_doc = {'wsf_gen':wsf_ori_gen_google, 
              'wsf_dom':wsf_ori_dom_google, 
              'rrf_gen':rrf_ori_gen_google, 
              'rrf_dom':rrf_ori_dom_google}


# # Iterate through the combination
for k,v in google_doc.items():
    fuse_score_dict = v
    results_per_query = {}
    du_res_q_oai = {}
    print(k)
    for query_id in query_ids:
    
        top_m_candidate_names = [name for name in fuse_score_dict[query_id].keys()]
        M = len(top_m_candidate_names)
        top_m_candidate_contents = {name: df_corpus[df_corpus['id']==name]['content'].iloc[0] for name in top_m_candidate_names}


        # Generate the M x M similarity matrix for ONLY these M candidates
      
        sim_matrix_M_by_M = get_candidate_pairwise_similarity_dict(top_m_candidate_contents,doc_id_to_emb)
        
        # Use top_m_candidates_names as the consistent order
        num_m_docs = M
        all_doc_sims_M_by_M_array = np.zeros((num_m_docs, num_m_docs))
        for i, doc_name_i in enumerate(top_m_candidate_names):
            for j, doc_name_j in enumerate(top_m_candidate_names):
                # Ensure sim_matrix_M_by_M has all pairs for these M docs
                all_doc_sims_M_by_M_array[i, j] = sim_matrix_M_by_M[doc_name_i][doc_name_j]

        # Get the fused scores for the current query
        current_query_fused_scores = fuse_score_dict.get(query_id, {})
        if not current_query_fused_scores:
            print(f"No fused scores found for query ID: {query_id}. Skipping.")
            results_per_query[query_id] = []
            continue

        # Convert current query's fused scores to a NumPy array, aligning with all_unique_doc_names_in_corpus order
        fused_prior_probabilities_array = np.array([
            current_query_fused_scores.get(doc_name, 0.0) # Default to 0.0 if doc not in this query's top fusion
            for doc_name in top_m_candidate_names
        ])

        fused_prior_probabilities_array = normalize_scores(fused_prior_probabilities_array)
        
        # Perform Document Utility
        res, selected_docs_for_query = document_utility_dense(
            fused_prior_probabilities=fused_prior_probabilities_array,
            all_doc_sims=all_doc_sims_M_by_M_array,
            doc_names=list(top_m_candidate_names),
            k=10,         # Get top 10 documents
            alpha=0.7    # Adjust alpha for each query if needed, or keep constant
        )

        results_per_query[query_id] = selected_docs_for_query
        du_res_q_oai[query_id]=res

    temp_score = calculate_retrieval_metrics(du_res_q_oai,qrels,k_values=[1,5,10])
    save_result('scores','google','docutil_'+k,du_res_q_oai)
    save_result('results','google','docutil_'+k,temp_score)

