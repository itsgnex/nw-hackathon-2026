from flask import Flask, request, jsonify
from flask_cors import CORS
from bot import legal_bot_response, classify_intent
import traceback

from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
# Enable CORS so our frontend can reliably talk to this backend
CORS(app)  # Enable CORS for all routes

@app.route('/api/query', methods=['POST'])
def handle_query():
    """
    Primary endpoint for processing user queries. 
    Handles text questions and optional PDF uploads for context.
    """
    try:
        # Extract fields from the multi-part form request
        user_query = request.form.get("query", "").strip()
        uploaded_file = request.files.get("file")
        
        # Log the file object to see what we're dealing with in the server logs
        print("DEBUG uploaded_file:", uploaded_file)


        # Basic check: we need something to process
        if not user_query and not uploaded_file:
            return jsonify({'error': 'Query cannot be empty'}), 400

        # UX tweak: if they only send a file, they probably want it summarized
        if not user_query and uploaded_file:
            user_query = "Summarize this document."
            
        # Extract text from PDF if one was uploaded
        pdf_text = ""
        if uploaded_file:
            # Importing locally to keep the main scope clean and only load pypdf when needed
            from read_pdf import extract_text_from_pdf_file
            pdf_text = extract_text_from_pdf_file(uploaded_file)

        # Construct the final input for the LLM.
        # If there's document text, we wrap it with specific instructions to stay within context.
        llm_input = user_query
        if pdf_text:
            llm_input = (
                "You are given a document below.\n"
                "Answer the user's question using ONLY this document.\n\n"
                "DOCUMENT:\n"
                f"{pdf_text}\n\n"
                "QUESTION:\n"
                f"{user_query}"
            )

        # Determine the user's intent (Rent, Work, Immigration, or General)
        detected_cat = classify_intent(user_query)
        print(f"--- Detected Intent: {detected_cat.upper()} ---")

        # Routing based on content type and intent:
        if uploaded_file:
            # If a document is present, use the document-aware prompt flow
            response = legal_bot_response(llm_input, detected_cat)
        elif detected_cat == "other":
            # Fallback for general greetings or non-legal questions
            response = (
                "Hi! I'm your BC Legal Assistant. I can help you with specific questions "
                "about **Rent**, **Work**, or **Immigration** in British Columbia. "
                "You can also upload a document for analysis."
            )
        else:
            # Standard legal query using web-search context (via Tavily)
            response = legal_bot_response(user_query, detected_cat)


        print(f"\n[{detected_cat.upper()} BOT]:\n{response}")

        return jsonify({
            'success': True,
            'response': response,
            'category': detected_cat
        }), 200

    except Exception as e:
        # Standard error handling: log the full traceback and return a generic 500
        print(f"Error processing query: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': f'Server error: {str(e)}',
            'success': False
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check for monitoring/uptime"""
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    # Run the dev server. Make sure port 5000 is open.
    app.run(debug=True, host='localhost', port=5000)
