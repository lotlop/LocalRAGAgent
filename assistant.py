import ollama
import chromadb
import psycopg
from tqdm import tqdm
from colorama import Fore
from psycopg.rows import dict_row
import ast

# Initialize the ChromaDB client for vector database operations
client = chromadb.Client()

# Define the system prompt that sets the AI's behavior and context awareness
system_prompt = (
    'You are an AI assistant that has memory of every conversation you have ever had with this user.'
    'On every prompt from the user, the system has checked for any relevant messages you have had with the user.'
    'If any embedded previous conversations are attached, use them for context to responding to the user, '
    'if the context is relevant and useful to responding. If the recalled conversations are irrelevant, '
    'disregard speaking about them and respond normally as an AI assistant. Do not talk about recalling conversations. '
    'Just use any useful data from the previous conversations and respond normally as an intelligent AI assistant.'
)

# Initialize conversation history with system prompt
convo = [{'role': 'system', 'content': system_prompt}]

# PostgreSQL database connection parameters
DB_PARAMS = {
    'dbname': 'memory_agent',
    'user': 'admin',
    'password': '123423',
    'host': 'localhost',
    'port': '5432'
}

def connect_db():
    """
    Creates and returns a connection to the PostgreSQL database
    Returns:
        psycopg.Connection: Database connection object
    """
    conn = psycopg.connect(**DB_PARAMS)
    return conn

def fetch_conversations():
    """
    Retrieves all conversations from the database
    Returns:
        list: List of dictionaries containing conversation data
    """
    conn = connect_db()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute('SELECT * FROM conversations')
        conversations = cursor.fetchall()
    conn.close()
    return conversations

def store_conversations(prompt, response):
    """
    Stores a new conversation in the database
    Args:
        prompt (str): User's input
        response (str): AI's response
    """
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute(
            'INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, %s, %s)', 
            (prompt, response)
        )
        conn.commit()
    conn.close()

def remove_last_conversation():
    """
    Removes the most recent conversation from the database
    Used when the user wants to forget the last interaction
    """
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute('DELETE FROM conversations WHERE id = (SELECT MAX(id) FROM conversations)')
        cursor.commit()

def stream_response(prompt):
    """
    Generates and streams the AI's response to the user's prompt
    Args:
        prompt (str): User's input message
    """
    response = ''
    
    # Stream response from the assistant
    stream = ollama.chat(model='llama3.1', messages=convo, stream=True)
    print(Fore.LIGHTGREEN_EX + '\nASSISTANT: ')

    # Collect and print the response in real-time
    for chunk in stream:
        content = chunk['message']['content']
        response += content
        print(content, end='', flush=True)
    
    print('\n')
    # # Store the conversation and update conversation history
    # store_conversations(prompt=prompt, response=response)
    convo.append({'role': 'assistant', 'content': response})

def create_vector_db(conversations):
    """
    Creates or recreates the vector database from conversation history
    Args:
        conversations (list): List of conversation dictionaries to embed
    """
    vector_db_name = 'conversations'

    # Delete existing collection if it exists
    try:
        client.delete_collection(name=vector_db_name)
    except ValueError:
        pass

    # Create a new collection
    vector_db = client.create_collection(name=vector_db_name)

    # Add each conversation to the vector database with embeddings
    for c in conversations:
        serialized_convo = f'prompt: {c["prompt"]} response: {c["response"]}'
        response = ollama.embeddings(model='nomic-embed-text', prompt=serialized_convo)
        embedding = response['embedding']

        vector_db.add(
            ids=[str(c['id'])],
            embeddings=[embedding],
            documents=[serialized_convo]
        )

def retrieve_embeddings(queries, results_per_query=2):
    """
    Retrieves relevant embedded conversations based on search queries
    Args:
        queries (list): List of search queries
        results_per_query (int): Number of results to retrieve per query
    Returns:
        set: Set of relevant conversation embeddings
    """
    embeddings = set()

    for query in tqdm(queries, desc='Processing queries to vector database'):
        response = ollama.embeddings(model='nomic-embed-text', prompt=query)
        query_embedding = response['embedding']

        vector_db = client.get_collection(name='conversations')
        results = vector_db.query(query_embeddings=[query_embedding], n_results=results_per_query)
        best_embeddings = results['documents'][0]

        # Add only relevant embeddings based on classification
        for best in best_embeddings:
            if best not in embeddings:
                if 'yes' in classify_embeddings(query=query, context=best):
                    embeddings.add(best)
    
    return embeddings

def create_queries(prompt):
    """
    Generates search queries for finding relevant past conversations
    Args:
        prompt (str): User's current input
    Returns:
        list: List of generated search queries
    """
    query_msg = (
        'You are a first principle reasoning search query AI agent. '
        'Your list of search queries will be ran on an embedding database of all your conversations '
        'you have ever had with the user. With first principles create a Python list of queries to '
        'search the embeddings database for any data that would be necessary to have access to in '
        'order to correctly respond to the prompt. Your response must be a Python list with no syntax errors. '  
        'Do not explain anything and do not ever generate anything but a perfect syntax python list'
    )
    
    # Example conversation to guide query generation
    query_convo = [
        {'role': 'system', 'content': query_msg},
        {'role': 'user', 'content': 'Write an email to my car insurance company and create a pursuasive resquest for them to lower my insurance cost'},
        {'role': 'assistant', 'content': '["What is the users name?", "What is the users current auto insurance provider", "How much does the user currently pay"]'},
        {'role': 'user', 'content': 'how can I convert the speak function in my llama3 python voice assistant to use pttsx3 instead of pyttsx'},
        {'role': 'assistant', 'content': '["Llama3 voice assistant", "Python voice assistant", "OpenAI TTS", "openai speak"]'},
        {'role': 'user', 'content': prompt},
    ]

    # Generate and parse response
    response = ollama.chat(model='llama3.1', messages=query_convo)
    print(Fore.YELLOW + f'\nVector database queries: {response["message"]["content"]}\n')

    try:
        return ast.literal_eval(response['message']['content'])
    except:
        return [prompt]

def classify_embeddings(query, context):
    """
    Determines if a retrieved embedding is relevant to the current query
    Args:
        query (str): Search query
        context (str): Retrieved embedding context
    Returns:
        str: 'yes' if relevant, 'no' if not
    """
    classify_msg = (
        'You are an embedding classification AI agent. Your input will be a prompt and one embedded chunk of text. '
        'You will not respond as an AI assistant. You only respond "yes" or "no". '
        'Determine whether the context contains data that directly is related to the search query. '
        'If the context is seemingly exactly the query needs, resppond "yes" if it is anything but directly '
        'related respond "no". Do not respond "yes" unless the content is highly relevant to the search query'
    )

    classify_convo = (
        {'role': 'system', 'content': classify_msg},
        {'role': 'user', 'content': f'SEARCH QUERY: what is the users name? \n\nEMBEDDED CONTEXT: You are Jean-Luc, how can I help you today?'},
        {'role': 'assistant', 'content': 'yes'},
        {'role': 'user', 'content': f'SEARCH QUERY: Llama3 Python Voice Assistant? \n\nEMBEDDED CONTEXT: Siri is a voice assistant developed by Apple.'},
        {'role': 'assistant', 'content': 'no'},
        {'role': 'user', 'content': f'SEARCH QUERY: {query} \n\nEMBEDDED CONTEXT: {context}'}
    )

    response = ollama.chat(model='llama3.1', messages=classify_convo)
    return response['message']['content'].strip().lower()

def recall(prompt):
    """
    Main function to recall relevant past conversations for context
    Args:
        prompt (str): User's current input
    """
    queries = create_queries(prompt=prompt)
    embeddings = retrieve_embeddings(queries=queries)
    convo.append({'role': 'user', 'content': f'MEMORIES: {embeddings} \n\n USER PROMPT: {prompt}'})
    print(f'\n{len(embeddings)} message: response embeddings added for context.')

# Initialize the system by creating vector database from existing conversations
conversations = fetch_conversations()
create_vector_db(conversations=conversations)

# Main interaction loop
while True:
    prompt = input(Fore.WHITE + 'USER: \n')

    # Handle different command prefixes
    if prompt[:7].lower() == '/recall':
        # Explicitly recall memories for this prompt
        prompt = prompt[8:]
        recall(prompt=prompt)
        stream_response(prompt=prompt)
    elif prompt[:7].lower() == '/forget':
        # Remove the last conversation
        remove_last_conversation()
        convo = convo[:-2]
        print('\n')
    elif prompt[:9].lower() == '/memorise':
        # Store a memory without generating a response
        prompt = prompt[10:]
        store_conversations(prompt=prompt, response='Memory stored')
        print('\n')
    else:
        # Normal conversation flow
        convo.append({'role': 'user', 'content': prompt})
        stream_response(prompt=prompt)