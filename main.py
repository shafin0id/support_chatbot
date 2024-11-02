from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from fastapi.responses import HTMLResponse
import gradio as gr
import json
from langchain_huggingface import HuggingFaceEmbeddings
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import os
import re
import shutil
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import uvicorn
from pdfminer.high_level import extract_text as pdf_extract
from docx import Document

# Initialize NLTK data
def initialize_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('punkt', quiet=True)
        print("NLTK data downloaded successfully!")

initialize_nltk()

def create_semantic_chunks(text: str, max_chunk_size: int = 1000) -> List[str]:
    try:
        initialize_nltk()
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
                
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    except Exception as e:
        print(f"Error in create_semantic_chunks: {str(e)}")
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word)
            if current_size + word_size > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(word)
            current_size += word_size + 1
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables and constants
UPLOAD_DIR = Path("uploaded_files")
SAVE_DIR = Path("saved_state")
UPLOAD_DIR.mkdir(exist_ok=True)
SAVE_DIR.mkdir(exist_ok=True)

# Custom prompt templates
CONDENSE_QUESTION_PROMPT = """Given the following conversation history and a new question, rephrase the question to be standalone, while keeping its original meaning and context.

Chat History:
{chat_history}

New Question: {question}

Standalone question:"""

QA_PROMPT = """You are a helpful and knowledgeable customer support agent trained on specific documents; you don't share any information outside of your trained data from those specific documents. Your goal is to provide accurate, short, and simplified information based on your training data while maintaining a friendly and engaging conversation.

If the question is a general greeting or casual conversation (like "how are you?", "good morning", etc.), respond naturally and warmly without referring to the training data.

For information-seeking questions:
1. If you find relevant information in the context, provide a clear and accurate response based on that information.
2. If you're unsure or the information isn't in the context, politely explain that you don't have that specific information.

Context: {context}

Question: {question}

Conversation History: {chat_history}

Answer: Let me help you with that."""

def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))

def sanitize_filename(filename: str) -> str:
    clean_name = re.sub(r'[^a-zA-Z0-9\-_.]', '_', filename)
    if len(clean_name) > 200:
        name_parts = clean_name.rsplit('.', 1)
        if len(name_parts) > 1:
            clean_name = f"{name_parts[0][:190]}.{name_parts[1]}"
        else:
            clean_name = clean_name[:200]
    return clean_name

def save_state() -> None:
    try:
        if state.vector_store is not None:
            state.vector_store.save_local(str(SAVE_DIR / 'vector_store'))
        
        with open(SAVE_DIR / 'doc_info.json', 'w') as f:
            json.dump(state.doc_info, f)
        
        with open(SAVE_DIR / 'trained_docs.json', 'w') as f:
            json.dump(list(state.trained_docs), f)
    except Exception as e:
        print(f"Error saving state: {e}")

def load_saved_state() -> None:
    try:
        vector_store_path = SAVE_DIR / 'vector_store'
        if vector_store_path.exists():
            state.vector_store = FAISS.load_local(
                str(vector_store_path),
                embeddings=state.embeddings
            )
        
        if (SAVE_DIR / 'doc_info.json').exists():
            with open(SAVE_DIR / 'doc_info.json', 'r') as f:
                state.doc_info = json.load(f)
        
        if (SAVE_DIR / 'trained_docs.json').exists():
            with open(SAVE_DIR / 'trained_docs.json', 'r') as f:
                state.trained_docs = set(json.load(f))
    except Exception as e:
        print(f"Error loading saved state: {e}")

class GlobalState:
    def __init__(self):
        self.vector_store: Optional[FAISS] = None
        self.doc_info: Dict[str, Dict[str, Any]] = {}
        self.trained_docs: set = set()
        self.training_start_time: Optional[float] = None
        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer',
            input_key='question'
        )
        self.llm = Ollama(
            model="gemma2:2b",
            temperature=0.7
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )

state = GlobalState()

def extract_text_from_pdf(file_path: Path) -> str:
    try:
        return pdf_extract(str(file_path))
    except Exception as e:
        raise Exception(f"Error extracting PDF text: {str(e)}")

def extract_text_from_docx(file_path: Path) -> str:
    try:
        doc = Document(str(file_path))
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        raise Exception(f"Error extracting DOCX text: {str(e)}")

def preprocess_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\n\r]+', '\n', text)
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\2', text)
    return text.strip()

def extract_and_process_text(file_path: Path) -> str:
    if file_path.suffix.lower() == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif file_path.suffix.lower() == '.docx':
        text = extract_text_from_docx(file_path)
    elif file_path.suffix.lower() == '.txt':
        text = file_path.read_text(encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    return preprocess_text(text)

def generate_response(message: str) -> str:
    casual_patterns = [
        r'^hi\b|^hello\b|^hey\b',
        r'how are you',
        r'good (morning|afternoon|evening)',
        r'thanks|thank you',
    ]
    
    is_casual = any(re.search(pattern, message.lower()) for pattern in casual_patterns)
    
    if is_casual:
        responses = {
            'greeting': [
                "Hello! How can I help you today?",
                "Hi there! I'm ready to assist you.",
                "Hey! What can I do for you?"
            ],
            'how_are_you': [
                "I'm doing well, thank you for asking! How can I help you?",
                "I'm great, thanks! What can I assist you with?",
            ],
            'thanks': [
                "You're welcome! Let me know if you need anything else.",
                "Happy to help! Feel free to ask more questions."
            ]
        }
        
        if re.search(r'how are you', message.lower()):
            return np.random.choice(responses['how_are_you'])
        elif re.search(r'thanks|thank you', message.lower()):
            return np.random.choice(responses['thanks'])
        else:
            return np.random.choice(responses['greeting'])

    if not state.vector_store:
        return "I'm not trained on any documents yet. Please upload and train some documents first!"

    try:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=state.llm,
            retriever=state.vector_store.as_retriever(
                search_kwargs={"k": 5}
            ),
            memory=state.memory,
            condense_question_prompt=PromptTemplate.from_template(CONDENSE_QUESTION_PROMPT),
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate.from_template(QA_PROMPT)
            },
            return_source_documents=True,
            verbose=True
        )

        response = qa_chain({"question": message})
        return response['answer']

    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

def build_gradio_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # Create a ChatGPT with your own data...beta v0.65_dev_shafinoid
        Upload your documents, train the model, and start chatting!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                file_output = gr.File(
                    file_count="multiple",
                    label="Upload Documents",
                    file_types=[".pdf", ".docx", ".txt"]
                )
                train_button = gr.Button("🚀 Train Model", variant="primary")
                training_info = gr.JSON(label="Training Status")
                
                with gr.Accordion("Documents Info", open=False):
                    docs_info = gr.JSON(label="Trained Documents")
            
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=600)
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your message",
                        placeholder="Type your question here...",
                        show_label=False,
                        scale=4
                    )
                    send = gr.Button(
                        "Send", 
                        variant="primary",
                        scale=1
                    )

        def user_message(message: str, history: List) -> Tuple[str, List]:
            return "", history + [[message, None]]

        def bot_message(history: List) -> List:
            message = history[-1][0]
            response = generate_response(message)
            history[-1][1] = response
            return history

        def handle_file_upload(files: List[Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, int]]]:
            if not files:
                return None, None
            
            uploaded_files = []
            for file in files:
                try:
                    original_name = Path(file.name).stem
                    extension = Path(file.name).suffix
                    unique_id = str(uuid.uuid4())[:8]
                    
                    clean_name = sanitize_filename(original_name)
                    unique_filename = f"{clean_name}_{unique_id}{extension}"
                    
                    file_path = UPLOAD_DIR / unique_filename
                    UPLOAD_DIR.mkdir(exist_ok=True)
                    
                    shutil.copy2(file.name, file_path)
                    
                    if not file_path.exists():
                        raise FileNotFoundError(f"Failed to copy file to {file_path}")
                    
                    state.doc_info[unique_filename] = {
                        'filename': unique_filename,
                        'original_name': file.name,
                        'upload_time': datetime.now().isoformat(),
                        'trained': False,
                        'file_path': str(file_path)
                    }
                    uploaded_files.append(unique_filename)
                    
                except Exception as e:
                    print(f"Error processing file {file.name}: {e}")
                    continue
            
            if not uploaded_files:
                return {"message": "No files were successfully uploaded"}, get_docs_info()
            
            return (
                {"message": f"Successfully uploaded {len(uploaded_files)} files", "files": uploaded_files},
                get_docs_info()
            )

        def get_docs_info() -> Dict[str, int]:
            return {
                "Total Documents": len(state.doc_info),
                "Trained Documents": len(state.trained_docs),
                "Total Chunks": sum(info.get('num_chunks', 0) for info in state.doc_info.values()),
                "Total Characters": sum(info.get('char_count', 0) for info in state.doc_info.values())
            }

        def train_model(progress=gr.Progress()) -> Tuple[Dict[str, str], Dict[str, int]]:
            if not state.doc_info:
                return {"message": "No documents to train"}, get_docs_info()

            state.training_start_time = time.time()
            progress(0, desc="Starting training...")
            
            try:
                untrained_files = [f for f in state.doc_info.values() if not f['trained']]
                total_files = len(untrained_files)
                
                if total_files == 0:
                    return {"message": "No new documents to train"}, get_docs_info()

                all_chunks = []
                for i, info in enumerate(untrained_files):
                    try:
                        filename = info['filename']
                        progress((i + 1) / total_files, desc=f"Processing {filename}")
                        file_path = Path(info['file_path'])
                        if not file_path.exists():
                            raise FileNotFoundError(f"File not found: {file_path}")
                        
                        text = extract_and_process_text(file_path)
                        chunks = create_semantic_chunks(text)
                        all_chunks.extend(chunks)
                        
                        state.doc_info[filename]['trained'] = True
                        state.doc_info[filename]['char_count'] = len(text)
                        state.doc_info[filename]['num_chunks'] = len(chunks)
                    except Exception as e:
                        print(f"Error processing file {filename}: {e}")
                        continue

                if all_chunks:
                    progress(0.9, desc="Creating vector store...")
                    new_vector_store = FAISS.from_texts(
                        texts=all_chunks,
                        embedding=state.embeddings
                    )

                    if state.vector_store is None:
                        state.vector_store = new_vector_store
                    else:
                        state.vector_store.merge_from(new_vector_store)

                    state.trained_docs.update([f['filename'] for f in untrained_files])
                    save_state()

                training_time = time.time() - state.training_start_time if state.training_start_time else 0
                return {
                    "message": f"Training completed in {format_time(training_time)}",
                    "time": format_time(training_time)
                }, get_docs_info()

            except Exception as e:
                return {"message": f"Error during training: {str(e)}"}, get_docs_info()

        # Set up event handlers
        msg.submit(user_message, [msg, chatbot], [msg, chatbot]).then(
            bot_message, chatbot, chatbot
        )
        send.click(user_message, [msg, chatbot], [msg, chatbot]).then(
            bot_message, chatbot, chatbot
        )
        file_output.upload(handle_file_upload, file_output, [training_info, docs_info])
        train_button.click(train_model, None, [training_info, docs_info])

    return interface

# Load saved state on startup
load_saved_state()

# Create Gradio interface
demo = build_gradio_interface()

# Mount Gradio app in FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <meta http-equiv="refresh" content="0;url=/gradio" />
        </head>
    </html>
    """

# New endpoint to create embedded code
@app.get("/create-embedded-code")
async def create_embedded_code():
    embedded_code = """
    <script>
    (function() {
        var chatButton = document.createElement('div');
        chatButton.innerHTML = '💬';
        chatButton.style.position = 'fixed';
        chatButton.style.bottom = '20px';
        chatButton.style.right = '20px';
        chatButton.style.width = '50px';
        chatButton.style.height = '50px';
        chatButton.style.borderRadius = '25px';
        chatButton.style.backgroundColor = '#007bff';
        chatButton.style.color = 'white';
        chatButton.style.fontSize = '24px';
        chatButton.style.display = 'flex';
        chatButton.style.justifyContent = 'center';
        chatButton.style.alignItems = 'center';
        chatButton.style.cursor = 'pointer';
        chatButton.style.boxShadow = '0 2px 10px rgba(0,0,0,0.2)';
        chatButton.style.zIndex = '9999';

        var chatInterface = document.createElement('iframe');
        chatInterface.src = 'http://shafinoid.local/gradio';
        chatInterface.style.position = 'fixed';
        chatInterface.style.bottom = '80px';
        chatInterface.style.right = '20px';
        chatInterface.style.width = '350px';
        chatInterface.style.height = '500px';
        chatInterface.style.border = 'none';
        chatInterface.style.borderRadius = '10px';
        chatInterface.style.boxShadow = '0 5px 15px rgba(0,0,0,0.3)';
        chatInterface.style.display = 'none';
        chatInterface.style.zIndex = '10000';

        document.body.appendChild(chatButton);
        document.body.appendChild(chatInterface);

        chatButton.addEventListener('click', function() {
            if (chatInterface.style.display === 'none') {
                chatInterface.style.display = 'block';
                chatButton.innerHTML = '✕';
            } else {
                chatInterface.style.display = 'none';
                chatButton.innerHTML = '💬';
            }
        });
    })();
    </script>
    """
    return {"embedded_code": embedded_code}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
