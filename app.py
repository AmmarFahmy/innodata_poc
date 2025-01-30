import streamlit as st
import sqlite3
import hashlib
import PyPDF2
from openai import OpenAI
import io
import json
from pdf2image import convert_from_bytes
from PIL import Image

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Create users table if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, role TEXT)''')
    
    # Add default users if they don't exist
    c.execute("SELECT * FROM users WHERE username='admin'")
    if not c.fetchone():
        admin_pass = hashlib.sha256("admin".encode()).hexdigest()
        c.execute("INSERT INTO users VALUES (?, ?, ?)", ("admin", admin_pass, "admin"))
    
    c.execute("SELECT * FROM users WHERE username='user'")
    if not c.fetchone():
        user_pass = hashlib.sha256("user".encode()).hexdigest()
        c.execute("INSERT INTO users VALUES (?, ?, ?)", ("user", user_pass, "user"))
    
    # Create PDF metadata table if not exists (without dropping)
    c.execute('''CREATE TABLE IF NOT EXISTS pdf_metadata
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  file_name TEXT,
                  page_number INTEGER,
                  page_content TEXT,
                  page_image BLOB,
                  keywords TEXT,
                  embeddings TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()

# Verify login credentials
def verify_login(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT role FROM users WHERE username=? AND password=?", (username, hashed_password))
    result = c.fetchone()
    
    conn.close()
    if result:
        return result[0]
    return None

# Admin Dashboard
def show_admin_dashboard():
    st.title("Admin Dashboard")
    st.write("Welcome, Admin!")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=['pdf'], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for pdf_file in uploaded_files:
            st.write(f"Processing: {pdf_file.name}")
            try:
                process_pdf(pdf_file, pdf_file.name)
                st.success(f"Successfully processed {pdf_file.name}")
            except Exception as e:
                st.error(f"Error processing {pdf_file.name}: {str(e)}")
    
    # Display existing PDF metadata with images
    st.subheader("Processed PDF Pages")
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT file_name, page_number, keywords, page_image FROM pdf_metadata ORDER BY file_name, page_number")
    results = c.fetchall()
    
    if results:
        for result in results:
            with st.expander(f"{result[0]} - Page {result[1]}"):
                # Display image
                if result[3]:  # if page_image exists
                    image = Image.open(io.BytesIO(result[3]))
                    st.image(image, caption=f"Page {result[1]}")
                st.write("Keywords:", result[2])
    else:
        st.info("No PDF files have been processed yet.")
    
    conn.close()

def calculate_keyword_similarity(query_keywords, stored_keywords):
    # Convert comma-separated strings to sets of words
    query_set = set(query_keywords.lower().split(','))
    stored_set = set(stored_keywords.lower().split(','))
    
    # Calculate Jaccard similarity
    intersection = len(query_set.intersection(stored_set))
    union = len(query_set.union(stored_set))
    
    return intersection / union if union > 0 else 0

def get_answer_from_gpt4(query, page_content):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    prompt = f"""Based on the following page content, please answer the user's question.
                If the answer cannot be found in the content, please say so.
                If user greets, please respond with a friendly greeting with a smiley face and a coversational tone. 
                If user asks for help, please respond with a friendly greeting. 
                
                User Question: {query}
                
                Page Content:
                {page_content}
                """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    if response.choices[0].message.content:
        print(response.choices[0].message.content)
    return response.choices[0].message.content.strip()

def show_user_dashboard():
    st.title("Chat Interface")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "image" in message:
                st.image(message["image"], caption=message["image_caption"])
    
    # Accept user input
    if prompt := st.chat_input("Ask your question here"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate keywords for the query
        query_keywords = get_keywords_from_gpt4(prompt)
        
        # Find the most relevant page from the database
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT page_content, keywords, page_image, file_name, page_number FROM pdf_metadata")
        results = c.fetchall()
        conn.close()
        
        # Calculate similarity scores and find the best match
        best_score = -1
        best_match = None
        
        for result in results:
            page_content, stored_keywords, page_image, file_name, page_number = result
            similarity_score = calculate_keyword_similarity(query_keywords, stored_keywords)
            
            if similarity_score > best_score:
                best_score = similarity_score
                best_match = result
        
        if best_match:
            # Generate answer using GPT-4
            answer = get_answer_from_gpt4(prompt, best_match[0])  # best_match[0] is page_content
            
            # Display assistant response with the relevant page image
            with st.chat_message("assistant"):
                st.markdown(answer)
                if best_match[2]:  # If page_image exists
                    image = Image.open(io.BytesIO(best_match[2]))
                    st.image(image, caption=f"Reference: {best_match[3]} - Page {best_match[4]}")
            
            # Add to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "image": best_match[2],
                "image_caption": f"Reference: {best_match[3]} - Page {best_match[4]}"
            })
        else:
            with st.chat_message("assistant"):
                st.markdown("I couldn't find any relevant information in the database.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I couldn't find any relevant information in the database."
            })

def get_keywords_from_gpt4(content):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    prompt = f"""Extract 15 relevant keywords from the following page_content. 
                 Return only the keywords as a comma-separated list:
                 <page_content>
                 {content}
                 </page_content>
                 """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    keywords = response.choices[0].message.content.strip()
    return keywords

def get_embeddings(text):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    if json.dumps(response.data[0].embedding):
        print("Embedding generated successfully")
    return json.dumps(response.data[0].embedding)  # Convert embedding vector to JSON string for storage

def process_pdf(pdf_file, file_name):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Check if file already exists in database
    c.execute("SELECT DISTINCT file_name, page_number FROM pdf_metadata WHERE file_name = ?", (file_name,))
    existing_pages = {(row[0], row[1]) for row in c.fetchall()}
    
    if existing_pages:
        st.warning(f"File {file_name} already exists in database. Skipping processing.")
        conn.close()
        return
    
    # Convert PDF file to bytes for pdf2image
    pdf_bytes = pdf_file.read()
    pdf_file.seek(0)  # Reset file pointer for PyPDF2
    
    # Convert PDF pages to images
    images = convert_from_bytes(pdf_bytes)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    for page_num in range(len(pdf_reader.pages)):
        with st.status(f"Processing page {page_num + 1} of {file_name}..."):
            # Check if this specific page already exists
            if (file_name, page_num + 1) in existing_pages:
                st.info(f"Page {page_num + 1} already exists. Skipping...")
                continue
                
            # Extract page content
            page = pdf_reader.pages[page_num]
            page_content = page.extract_text()
            
            # Convert page image to bytes
            img = images[page_num]
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Get keywords using GPT-4
            st.write("Generating keywords...")
            keywords = get_keywords_from_gpt4(page_content)
            
            # Get embeddings for page content
            st.write("Generating embeddings...")
            embeddings = get_embeddings(page_content)
            
            # Store in database
            c.execute("""
                INSERT INTO pdf_metadata (file_name, page_number, page_content, page_image, keywords, embeddings)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (file_name, page_num + 1, page_content, img_byte_arr, keywords, embeddings))
            
            conn.commit()
            st.success(f"Page {page_num + 1} processed successfully")
    
    conn.close()

def main():
    st.set_page_config(page_title="POC", page_icon="📃")
    
    # Initialize the database
    init_db()
    
    # Session state initialization
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.role = None

    # If not logged in, show login form
    if not st.session_state.logged_in:
        st.title("Login")
        
        # Create login form
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                role = verify_login(username, password)
                if role:
                    st.session_state.logged_in = True
                    st.session_state.role = role
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    # Show appropriate dashboard based on role
    else:
        if st.session_state.role == "admin":
            show_admin_dashboard()
        else:
            show_user_dashboard()
        
        # Add logout button
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.role = None
            st.rerun()

if __name__ == "__main__":
    main() 