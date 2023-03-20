from flask import request, jsonify, send_from_directory
from simple_chat_pdf.components.data_ingestion import DataIngestion

data_ingestion = DataIngestion()

def configure_routes(app):
    @app.route('/parse-pdf', methods=['POST'])
    def parse_pdf():
        pdf_file = request.files.get('pdf')
        data_ingestion.parse_pdf(pdf_file)
        return jsonify({"status": "success", "message": "PDF parsed successfully"})

    @app.route('/')
    def root():
        pdf_file = "./case_studies.pdf"
        status = data_ingestion.parse_pdf(pdf_file)
        return jsonify({"status": "success", "message": "PDF parsed successfully", "result": status})
