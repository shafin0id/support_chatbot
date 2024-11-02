# Custom Document-Based Support Chatbot with Gemma 2B

<p align="center">
  <img src="https://github.com/user-attachments/assets/d7a4102f-1a25-47b5-a21a-dbfed3a6f9c3" alt="Custom Chatbot Interface" width="800">
</p>

Create a powerful, customizable chatbot that learns from your documents. This project leverages FastAPI, Gradio, LangChain, and Ollama with the Gemma 2B model to deliver an intelligent question-answering system based on your data.

## Features

- **Document Upload**: Support for PDF, DOCX, and TXT files
- **Semantic Text Processing**: Intelligent chunking of document content using NLTK
- **Vector Storage**: Efficient similarity search using FAISS
- **Language Model Integration**: Utilizes Ollama with Gemma 2B for natural language understanding and generation
- **Conversational AI**: Context-aware responses using ConversationalRetrievalChain
- **User-Friendly Interface**: Easy-to-use Gradio web interface
- **Embeddable Chat Widget**: Generate code for embedding the chat on any website
- **State Management**: Save and load trained models
- **FastAPI Backend**: Robust API support with CORS enabled
- **Progress Tracking**: Visual feedback during document processing and training
- **Casual Conversation Handling**: Detects and responds to greetings and thanks
- **Document Information Tracking**: Keeps metadata for uploaded and trained documents

## Installation

1. Clone the repository:
git clone https://github.com/yourusername/custom-document-chatbot.git cd custom-document-chatbot
Copy

2. Create a virtual environment:
python -m venv venv source venv/bin/activate # On Windows use venv\Scripts\activate
Copy

3. Install the required packages:
pip install -r requirements.txt
Copy

4. Install Ollama and the Gemma 2B model:
Install Ollama (instructions may vary based on your OS)

curl https://ollama.ai/install.sh | sh
Pull the Gemma 2B model

ollama pull gemma2:2b
Copy

5. Run the application:
python main.py
Copy

6. Open your browser and navigate to `http://localhost:8000` to access the chatbot interface.

## Usage

1. **Upload Documents**: Use the file upload section to add your PDF, DOCX, or TXT files.
2. **Train Model**: Click the "Train Model" button to process the uploaded documents.
3. **Chat**: Once training is complete, use the chat interface to ask questions about your documents.

## Embedding the Chat Widget

To embed the chat widget on your website, use the following code:

```html
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
 chatInterface.src = 'http://yourdomain.com/gradio';
 chatInterface.style.position = 'fixed';
 chatInterface
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
Replace 'http://yourdomain.com/gradio' with the actual URL where your chatbot is hosted.
Development

This project uses:
FastAPI for the backend server
Gradio for the user interface
LangChain for document processing and chain management
FAISS for vector storage
Ollama with Gemma 2B for language modeling
NLTK for text processing
To contribute or modify:
Fork the repository
Create a new branch for your feature
Implement your changes
Submit a pull request
License

This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Thanks to the Ollama team for providing the Gemma 2B model
LangChain for their excellent library for building LLM applications
Gradio for the easy-to-use UI components
