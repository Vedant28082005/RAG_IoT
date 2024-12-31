from flask import Flask, request, render_template, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Load the PDF
pdf_reader = PdfReader("example2.pdf")
text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

# Split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Adjust size as needed
    chunk_overlap=50  # Overlap to maintain context
)
chunks = text_splitter.split_text(text)

# Load a free embeddings model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Free model

# Generate embeddings for the chunks
embeddings = model.encode(chunks)

# Create a FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
index.add(np.array(embeddings))

# Save FAISS index locally (optional)
faiss.write_index(index, "faiss_index")

app = Flask(__name__)

@app.route('/esp32', methods=['GET'])  # Use GET for data retrieval
def esp32():
    # Get temperature and humidity from the query parameters
    temperature = request.args.get('temperature')
    humidity = request.args.get('humidity')
    
    

    # Validate that both parameters are provided
    if temperature is not None and humidity is not None:
        try:
            # Convert to float for validation and processing
            temperature = float(temperature)
            humidity = float(humidity)
            
            query = (f'Temperature and Humidity of the field is {temperature} and {humidity} respectively. Please give your review for the field')
            
            query_embedding = model.encode([query])
            # Search FAISS for the top matches
            k = 5  # Number of chunks to retrieve
            distances, indices = index.search(np.array(query_embedding), k)
            
            retrieved_chunks = [chunks[i] for i in indices[0]]
            
            generator = pipeline('text-generation', model='bigscience/bloom-560m', device='cpu')
            context = " ".join(retrieved_chunks)
            response = generator(
            f"Context: {context}\n\nQuestion: {query}\nAnswer:",
            max_new_tokens=50,  # Number of tokens to generate
            truncation=True     # Truncate the input if it exceeds the max length
            )
            
            return jsonify(status="success", message = response[0]['generated_text']), 200
            
            
            
        except ValueError:
            # If conversion to float fails, return 400 Bad Request
            return jsonify(status="error", message="Invalid temperature or humidity value"), 400
    else:
        # If either parameter is missing, return 400 Bad Request
        return jsonify(status="error", message="Both 'temperature' and 'humidity' are required"), 400

if __name__ == '__main__':
    app.run(debug=True)