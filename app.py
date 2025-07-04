import os
import torch # Import torch - This is for PyTorch
from transformers import AutoTokenizer, AutoModel # AutoModel will load PyTorch model if torch is imported
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from flask import Flask, request, render_template, session, jsonify
import uuid
from datetime import datetime
import ssl
from flask_session import Session # Import Flask-Session

# --- Perbaikan NLTK Resource Download ---
try:
    # Disable SSL verification untuk NLTK download jika diperlukan
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Download punkt tokenizer dengan cara yang benar
    print("Mengunduh resource NLTK..." )
    nltk.download('punkt', quiet=True)
    
    # Test apakah punkt tokenizer berfungsi
    try:
        sent_tokenize("Ini adalah kalimat test. Ini kalimat kedua.", language='indonesian')
        print("NLTK punkt tokenizer berhasil dimuat.")
    except Exception as e:
        print(f"Warning: Gagal memuat punkt tokenizer untuk bahasa Indonesia: {e}")
        print("Menggunakan tokenizer default...")
        
except Exception as e:
    print(f"Error saat mengunduh resource NLTK: {e}")

# --- Konfigurasi Model IndoBERT (Embeddings) ---
print("Memuat tokenizer dan model IndoBERT untuk embeddings...")
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
# Load model for PyTorch and set to evaluation mode
embedding_model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
embedding_model.eval() # Set PyTorch model to evaluation mode (disables dropout, etc.)
# Check for GPU and move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch device setup
embedding_model.to(device) # Move PyTorch model to GPU/CPU
print(f"Menggunakan perangkat: {device}")


# --- In-memory cache for processed document data (chunks and embeddings) ---
# This dictionary will store processed document data to avoid re-computation.
# Key: unique_file_id (UUID), Value: {'chunks': [], 'embeddings': np.array([])}
document_data_cache = {}

# --- Fungsi Ekstraksi Teks dari PDF ---
def extract_text_from_pdf(pdf_path):
    """
    Extracts text content from a PDF file.
    Args:
        pdf_path (str): The path to the PDF file.
    Returns:
        str: The extracted text, or None if an error occurs.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            extracted_page_text = page.extract_text()
            if extracted_page_text:
                text += extracted_page_text + " "
        # Clean up text: replace newlines, non-breaking spaces, and normalize whitespace
        text = text.replace('\n', ' ').replace('\xa0', ' ').strip()
        text = ' '.join(text.split())
        return text
    except Exception as e:
        print(f"Error saat mengekstrak teks dari PDF: {e}")
        return None

# --- Fungsi untuk Membagi Dokumen menjadi Chunks ---
def chunk_document(text, chunk_size=300, overlap=50):
    """
    Splits a given text into smaller chunks with a specified overlap.
    Args:
        text (str): The input text.
        chunk_size (int): The maximum number of words per chunk.
        overlap (int): The number of words to overlap between consecutive chunks.
    Returns:
        list: A list of text chunks.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# --- Fungsi untuk Mendapatkan Embeddings Chunks ---
def get_chunk_embeddings(chunks, tokenizer, model, batch_size=32):
    """
    Generates embeddings for a list of text chunks using the provided tokenizer and PyTorch model.
    Args:
        chunks (list): A list of text chunks.
        tokenizer: The pre-trained tokenizer.
        model: The pre-trained PyTorch model.
        batch_size (int): The number of chunks to process in each batch.
    Returns:
        np.ndarray: A NumPy array of embeddings.
    """
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        # Tokenize for PyTorch, move to device
        inputs = tokenizer(batch_chunks, return_tensors='pt', padding=True, truncation=True, max_length=512) # return_tensors='pt' ensures PyTorch tensors
        inputs = {k: v.to(device) for k, v in inputs.items()} # Move inputs to PyTorch device (CPU/GPU)

        with torch.no_grad(): # Disable gradient calculation for inference (PyTorch specific)
            outputs = model(**inputs)
        
        # Get the [CLS] token embedding (first token of the last hidden state)
        # Move to CPU and convert to NumPy array for scikit-learn cosine_similarity
        all_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy()) # .cpu().numpy() is PyTorch specific
    
    if all_embeddings:
        return np.vstack(all_embeddings)
    else:
        return np.array([])

# --- Fungsi Sederhana untuk Membuat Ringkasan Ekstraktif ---
def extractive_summarize(text, max_sentences=3):
    """
    Generates a simple extractive summary based on sentence length and position.
    Args:
        text (str): The input text to summarize.
        max_sentences (int): The maximum number of sentences in the summary.
    Returns:
        str: The generated summary.
    """
    try:
        # Attempt to use Indonesian specific tokenizer, fallback to default, then simple split
        try:
            sentences = sent_tokenize(text, language='indonesian')
        except:
            try:
                sentences = sent_tokenize(text)
            except:
                sentences = text.split('. ')
                sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return ' '.join(sentences)
        
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            # Skip very short sentences
            if len(sentence.split()) < 5:
                continue
                
            score = len(sentence.split())  # Score by length
            if i < 3:  # Boost for initial sentences
                score *= 1.5
            sentence_scores.append((score, sentence))
        
        # Fallback if no sentences meet criteria
        if not sentence_scores:
            sentence_scores = [(len(sentence.split()), sentence) for sentence in sentences]
        
        # Sort by score and take the best ones
        sentence_scores.sort(key=lambda x: x[0], reverse=True) # Sort by score
        top_sentences = [sentence for score, sentence in sentence_scores[:max_sentences]]
        
        # Preserve original order of top sentences
        top_sentences_in_order = [s for s in sentences if s in top_sentences]
        
        return ' '.join(top_sentences_in_order)
        
    except Exception as e:
        print(f"Error dalam membuat ringkasan: {e}")
        # Robust fallback: take first N sentences or a truncated part of the text
        try:
            sentences = text.split('. ')
            if len(sentences) >= max_sentences:
                return '. '.join(sentences[:max_sentences]) + ('.' if not sentences[max_sentences-1].endswith('.') else '')
            else:
                return text[:500] + "..." if len(text) > 500 else text
        except:
            return "Gagal membuat ringkasan."


# --- FUNGSI BARU: Menjawab Pertanyaan ---
def answer_question_from_pdf(pdf_id, question):
    """
    Answers a question based on the content of a previously processed PDF.
    Args:
        pdf_id (str): The unique ID of the processed PDF in the cache.
        question (str): The question to answer.
    Returns:
        str: The answer or an error message.
    """
    # Retrieve pre-computed data from cache
    processed_data = document_data_cache.get(pdf_id)
    if not processed_data:
        return "Maaf, dokumen ini tidak lagi tersedia dalam cache. Harap unggah ulang."

    chunks = processed_data['chunks']
    chunk_embeddings = processed_data['embeddings']

    if not chunks or chunk_embeddings.size == 0:
        return "Dokumen terlalu pendek atau kosong untuk dianalisis."

    # 1. Dapatkan embedding pertanyaan
    question_embedding = get_chunk_embeddings([question], tokenizer, embedding_model)
    
    if question_embedding.size == 0:
        return "Gagal menghasilkan representasi pertanyaan."

    # 2. Cari chunk paling relevan
    # Ensure dimensions match for cosine_similarity (1, D) vs (N, D)
    similarity_scores = cosine_similarity(question_embedding.reshape(1, -1), chunk_embeddings).flatten()
    
    # Ambil top 2 chunks untuk konteks yang lebih baik
    # Handle cases where there are fewer than 2 chunks
    num_top_chunks = min(2, len(chunks))
    top_indices = np.argsort(similarity_scores)[-num_top_chunks:][::-1]
    
    if not top_indices.size:
        return "Tidak ada konteks yang relevan ditemukan dalam dokumen."

    relevant_contexts = [chunks[i] for i in top_indices]
    combined_context = " ".join(relevant_contexts)

    print(f"Konteks yang paling relevan ditemukan (similarity: {similarity_scores[top_indices[0]]:.3f}):\n{combined_context[:300]}...")

    # For demo, return the relevant context with some formatting
    # In a real RAG system, you'd feed this context AND the question to a larger LLM
    return f"Berdasarkan analisis dokumen, berikut informasi yang relevan dengan pertanyaan Anda:\n\n{combined_context}\n\n(Skor relevansi: {similarity_scores[top_indices[0]]:.3f})"

# --- Konfigurasi Aplikasi Flask ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.secret_key = 'supersecretkey_for_flash_messages_and_session'

# Konfigurasi session menggunakan Flask-Session
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 jam
Session(app) # Initialize Flask-Session

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    """Checks if a file's extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles file uploads and question answering."""
    if request.method == 'POST':
        # Handle file upload
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file and allowed_file(file.filename):
                # Generate unique filename to avoid conflicts and for cache key
                unique_filename = f"{uuid.uuid4()}_{file.filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                
                try:
                    # --- Process the document immediately upon upload ---
                    full_text = extract_text_from_pdf(file_path)
                    if not full_text:
                        return jsonify({
                            'status': 'error',
                            'message': 'Gagal membaca teks dari dokumen. Pastikan PDF tidak kosong atau rusak.'
                        })
                    
                    chunks = chunk_document(full_text)
                    if not chunks:
                            return jsonify({
                                'status': 'error',
                                'message': 'Dokumen terlalu pendek atau kosong untuk dianalisis.'
                            })

                    chunk_embeddings = get_chunk_embeddings(chunks, tokenizer, embedding_model)
                    if chunk_embeddings.size == 0:
                        return jsonify({
                            'status': 'error',
                            'message': 'Gagal menghasilkan representasi dokumen. Dokumen mungkin terlalu singkat atau memiliki masalah pemrosesan.'
                        })
                    
                    summary = extractive_summarize(full_text)
                    
                    # Store processed data in the in-memory cache
                    document_data_cache[unique_filename] = {
                        'chunks': chunks,
                        'embeddings': chunk_embeddings
                    }
                    print(f"Dokumen '{unique_filename}' diproses dan disimpan di cache. Ukuran cache: {len(document_data_cache)}")

                    # Store relevant info in session
                    session['current_pdf_id'] = unique_filename # Use UUID as ID
                    session['current_pdf_name'] = file.filename
                    session['upload_time'] = datetime.now().isoformat()
                    session['document_summary'] = summary
                    session['document_preview'] = full_text[:500] + "..." if len(full_text) > 500 else full_text
                    session['chat_history'] = []  # Reset chat history for new document
                    session.modified = True # Mark session as modified
                    
                    return jsonify({
                        'status': 'success',
                        'message': 'File berhasil diunggah dan diproses!',
                        'filename': file.filename,
                        'summary': summary,
                        'preview': session['document_preview']
                    })
                except Exception as e:
                    print(f"Error during document processing: {e}")
                    # Clean up the uploaded file if processing fails
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    return jsonify({
                        'status': 'error',
                        'message': f'Terjadi kesalahan saat memproses dokumen: {str(e)}'
                    })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Format file tidak didukung. Harap unggah file PDF.'
                })
        
        # Handle question submission
        elif 'question_text' in request.form and request.form['question_text'] != '':
            question = request.form['question_text'].strip()
            
            # Check if a document has been uploaded and processed
            if 'current_pdf_id' not in session:
                return jsonify({
                    'status': 'error',
                    'message': 'Harap unggah dokumen PDF terlebih dahulu sebelum bertanya.'
                })
            
            pdf_id = session['current_pdf_id']
            
            # Check if processed data is still in cache
            if pdf_id not in document_data_cache:
                # This could happen if the server restarted or cache was cleared
                # Clean session info related to this document
                session.pop('current_pdf_id', None)
                session.pop('current_pdf_name', None)
                session.pop('document_summary', None)
                session.pop('document_preview', None)
                session.pop('chat_history', None)
                session.modified = True
                return jsonify({
                    'status': 'error',
                    'message': 'Dokumen tidak ditemukan dalam cache. Harap unggah ulang dokumen.'
                })
            
            try:
                answer = answer_question_from_pdf(pdf_id, question)
                
                # Update chat history
                if 'chat_history' not in session:
                    session['chat_history'] = []
                
                session['chat_history'].append({
                    'type': 'question',
                    'content': question,
                    'timestamp': datetime.now().isoformat()
                })
                session['chat_history'].append({
                    'type': 'answer',
                    'content': answer,
                    'timestamp': datetime.now().isoformat()
                })
                session.modified = True
                
                return jsonify({
                    'status': 'success',
                    'answer': answer
                })
                
            except Exception as e:
                print(f"Terjadi kesalahan saat memproses pertanyaan: {str(e)}") # Log error for debugging
                return jsonify({
                    'status': 'error',
                    'message': f"Terjadi kesalahan saat memproses pertanyaan: {str(e)}"
                })

    # GET request - render template
    return render_template('index.html')

@app.route('/clear_session', methods=['POST'])
def clear_session():
    """
    Endpoint to clear the current session and remove the uploaded file from disk.
    Also removes the document data from the in-memory cache.
    """
    try:
        pdf_id_to_remove = session.get('current_pdf_id')
        pdf_name_to_remove = session.get('current_pdf_name')

        # Remove from in-memory cache
        if pdf_id_to_remove and pdf_id_to_remove in document_data_cache:
            del document_data_cache[pdf_id_to_remove]
            print(f"Dokumen '{pdf_id_to_remove}' dihapus dari cache. Ukuran cache: {len(document_data_cache)}")

        # Remove the file from disk if it exists and matches the current session's file
        if pdf_id_to_remove and pdf_name_to_remove:
            # Construct the path to the uploaded file using the unique filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_id_to_remove)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File '{file_path}' berhasil dihapus.")
        
        session.clear() # Clear all session data
        print("Session berhasil dibersihkan.")
        
        return jsonify({
            'status': 'success',
            'message': 'Session berhasil dibersihkan'
        })
        
    except Exception as e:
        print(f"Error clearing session or file: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Terjadi kesalahan saat membersihkan session: {str(e)}'
        })

@app.route('/get_chat_history')
def get_chat_history():
    """
    Endpoint to retrieve the current chat history and document information.
    """
    chat_history = session.get('chat_history', [])
    document_info = {
        'has_document': 'current_pdf_id' in session,
        'filename': session.get('current_pdf_name', ''),
        'summary': session.get('document_summary', ''),
        'preview': session.get('document_preview', ''),
        'upload_time': session.get('upload_time', '')
    }
    
    return jsonify({
        'chat_history': chat_history,
        'document_info': document_info
    })

if __name__ == '__main__':
    # Flask-Session needs a directory to store session files if using filesystem backend
    if app.config['SESSION_TYPE'] == 'filesystem':
        if not os.path.exists('./flask_session'):
            os.makedirs('./flask_session')
        app.config['SESSION_FILE_DIR'] = './flask_session'
    
    app.run(debug=True, port=5000) # Run on port 5000