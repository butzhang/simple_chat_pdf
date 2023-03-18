from flask import request, jsonify, send_from_directory
from components.data_ingestion import DataIngestion
from components.user_question import UserQuestion
from components.vector_extraction import VectorExtraction
from components.answer_generator import AnswerGenerator

data_ingestion = DataIngestion(pinecone_api_key="5382dafe-fb42-4830-9dfa-622a0dd28e42", 
                               pinecone_environment="us-central1-gcp",
                               openai_api_key="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")


user_question = UserQuestion()
vector_extraction = VectorExtraction()
answer_generator = AnswerGenerator()

def configure_routes(app):
    @app.route('/parse-pdf', methods=['POST'])
    def parse_pdf():
        pdf_file = request.files.get('pdf')
        data_ingestion.parse_pdf(pdf_file)
        return jsonify({"status": "success", "message": "PDF parsed successfully"})

    @app.route('/ask-question', methods=['POST'])
    def ask_question():
        chat_history = request.json.get('chat_history')
        user_question = request.json.get('question')

        standalone_question = user_question.process_question(chat_history, user_question)
        embedding = vector_extraction.generate_embedding(standalone_question)
        relevant_docs = vector_extraction.query_vector_store(embedding)
        answer = answer_generator.generate_answer(standalone_question, relevant_docs)

        return jsonify({"status": "success", "answer": answer})

    @app.route('/')
    def root():
        pdf_file = "../case_studies.pdf"
        status = data_ingestion.parse_pdf(pdf_file)
        return jsonify({"status": "success", "message": "PDF parsed successfully", "result": status})


    @app.route('/uploads/<path:filename>')
    def download_file(filename):
      return send_from_directory('uploads', filename, as_attachment=True)