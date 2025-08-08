import openai
from tqdm import tqdm
import json

# Function to expand query domain focused
def expand_domain_focus(query: str, LLM_MODEL: str = '') -> str:
    """
    Generates an extensive elaboration and expansion of an Earth Science query
    using an OpenAI LLM, suitable for enhanced information retrieval.

    Args:
        query (str): The Earth Science query string to expand.

    Returns:
        str: A detailed, expanded version of the query, or an error message
             if the LLM response cannot be obtained.
    """
    prompt_file = './prompts/domain_focused_expansion.md'
    # Modified prompt for Earth Science domain-specific query expansion
    with open(prompt_file, 'r') as f:
        prompt_template = f.read()

    expansion_prompt = prompt_template.format(query=query)

    try:
        response = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful and knowledgeable Earth Science assistant."},
                {"role": "user", "content": expansion_prompt}
            ],
            temperature=0.2, 
            max_tokens=500   
        )

        expanded_query_text = response.choices[0].message.content.strip()
        return expanded_query_text

    except openai.APIError as e:
        print(f"OpenAI API Error during query expansion: {e}")
        return f"API Error: Could not expand query. {e}"
    except Exception as e:
        print(f"An unexpected error occurred during query expansion: {e}")
        return f"Internal Error: Could not expand query. {e}"

# Function to expand query general
def expand_general(query: str, LLM_MODEL: str = '') -> str:
    """
    Generates an extensive elaboration and expansion of a query
    using an OpenAI LLM, suitable for enhanced information retrieval.

    Args:
        query (str): The Earth Science query string to expand.

    Returns:
        str: A detailed, expanded version of the query, or an error message
             if the LLM response cannot be obtained.
    """
    prompt_file = './prompts/general_expansion.md'
    # Modified prompt for Earth Science domain-specific query expansion
    with open(prompt_file, 'r') as f:
        prompt_template = f.read()

    expansion_prompt = prompt_template.format(query=query)

    try:
        response = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": expansion_prompt}
            ],
            temperature=0.2, # Use a higher temperature for more creative and extensive expansion
            max_tokens=500   # Allow for a longer, more detailed expansion
        )

        expanded_query_text = response.choices[0].message.content.strip()
        return expanded_query_text

    except openai.APIError as e:
        print(f"OpenAI API Error during query expansion: {e}")
        return f"API Error: Could not expand query. {e}"
    except Exception as e:
        print(f"An unexpected error occurred during query expansion: {e}")
        return f"Internal Error: Could not expand query. {e}"

# Function to call the LLM API for query expansion
def query_expansion(df_queries, LLM_model):
    expanded_general = []
    expanded_domain = []

    print(f"Using OpenAI Model for expansion")

    for q in tqdm(df_queries['query']):
        expanded_domain_text = expand_domain_focus(q, LLM_model)
        expanded_domain.append(expanded_domain_text)  
        expanded_general_text = expand_general(q, LLM_model)
        expanded_general.append(expanded_general_text)

    df_queries['expanded_general'] = expanded_general
    df_queries['expanded_focus'] = expanded_domain

    return df_queries

# Function to embedd OPENAI
def get_embedding_openai(texts, openai_client,tokenizer,model="text-embedding-3-large"):
    texts =[json.dumps(text.replace("\n", " ")) for text in texts]
    success = False
    threshold = 6000
    count = 0
    cur_emb = None
    exec_count = 0
    while not success:
        exec_count += 1
        if exec_count>5:
            print('execute too many times')
            exit(0)
        try:
            emb_obj = openai_client.embeddings.create(input=texts, model=model).data
            cur_emb = [e.embedding for e in emb_obj]
            success = True
        except Exception as e:
            print(e)
            count += 1
            threshold -= 500
            if count>4:
                print('openai cut',count)
                exit(0)
            new_texts = []
            for t in texts:
                new_texts.append(cut_text_openai(text=t, tokenizer=tokenizer,threshold=threshold))
            texts = new_texts
    if cur_emb is None:
        raise ValueError("Fail to embed, openai")
    return cur_emb

# function to cut text if it exceed the threshold
def cut_text(text,tokenizer,threshold):
    text_ids = tokenizer(text)['input_ids']
    if len(text_ids) > threshold:
        text = tokenizer.decode(text_ids[:threshold])
    return text

def cut_text_openai(text,tokenizer,threshold=6000):
    token_ids = tokenizer.encode(text)
    if len(token_ids) > threshold:
        text = tokenizer.decode(token_ids[:threshold])
    return text



