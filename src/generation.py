import os
from openai import OpenAI

# Setup the API client
client = OpenAI(
    api_key="YOUR_API_KEY_HERE",
    base_url="https://api.groq.com/openai/v1"
)

# Get text for each chunk ID
def get_chunk_texts(retrieved_ids, chunks):
    # Map IDs to their text
    chunk_dict = {c["chunk_id"]: c["text"] for c in chunks}
    
    texts = []
    # Find text for each ID in the list
    for cid in retrieved_ids:
        if cid in chunk_dict:
            texts.append(chunk_dict[cid])
            
    return texts

# Use the AI to write an answer
def generate_answer(query, context_texts):
    # Combine all text pieces into one string
    context_str = "\n\n---\n\n".join(context_texts)
    
    # Set rules for the AI
    system_prompt = (
        "You are a helpful QA bot. "
        "Answer the question using ONLY the provided context. "
        "If context has no answer, say 'I do not know'. "
        "Briefly cite your sources."
    )
    
    # Prepare the question and context
    user_prompt = f"Context:\n{context_str}\n\nQuestion:\n{query}"
    
    try:
        # Ask the AI for the answer
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        
        # Return the AI text
        return response.choices[0].message.content
        
    except Exception as e:
        # Show error if API fails
        print(f"API Error: {e}")
        return "Error generating answer."

# Run the answer process for all queries
def run_generation_step(sampled_queries, chunks, top_k=5):
    print(f"  Starting LLM generation for {len(sampled_queries)} queries...")
    
    for q in sampled_queries:
        # Use top search results
        top_ids = q.get("colbert_retrieved_ids", [])[:top_k]
        
        # Get text for the search results
        context_texts = get_chunk_texts(top_ids, chunks)
        
        # Create the answer
        answer = generate_answer(q["query"], context_texts)
        
        # Store the answer
        q["generated_answer"] = answer
        
    print("  Generation step finished.")