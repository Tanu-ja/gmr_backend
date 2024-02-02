from flask import Flask, jsonify, request
import json
import requests
import os
import openai
from langdetect import detect
from dotenv import load_dotenv
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

project_folder = os.path.dirname(__file__)

load_dotenv(os.path.join(project_folder, '.env'))

ENV_ENABLE = os.getenv('ENV_ENABLE');
API_VERSION = os.getenv('API_VERSION')
API_KEY = os.getenv('API_KEY')
OPENAI_ENGINE = os.getenv('OPENAI_ENGINE')
OPENAI_URL = os.getenv('OPENAI_URL')
COGNITIVE_SEARCH_ENDPOINT = os.getenv('COGNITIVE_SEARCH_ENDPOINT')
COGNITIVE_SEARCH_KEY = os.getenv('COGNITIVE_SEARCH_KEY')
COGNITIVE_SEARCH_INDEX_NAME = os.getenv('COGNITIVE_SEARCH_INDEX_NAME')

def translate_text(text, target_language):
    openai.api_type = "azure"
    openai.api_base = "https://azdogropenaidev.openai.azure.com/"
    openai.api_version = "2023-07-01-preview"
    openai.api_key = os.getenv("API_KEY")

    message_text = [
        {"role":"system","content":"You are an AI assistant that translates text. You only respond with the translated text. You make sure that there are no spelling mistakes in your response."},
        {"role":"user","content":f"Translate the following test to, please make sure that structure and meaing of the sentense is not changed {target_language} : {text}"}]

    completion = openai.ChatCompletion.create(
    engine="gopgpt35",
    messages = message_text,
    temperature=0,
    max_tokens=800,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None,
    stream=False
    )
    

    return completion["choices"][0]["message"]["content"]

@app.route("/")
def index():
    return f"<center><h1>Flask App deployment on AZURE</h1></center"

@app.route("/get_response", methods=["POST"])
@cross_origin()
def get_response():
    url = OPENAI_URL

    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }
    user_input = request.get_json().get("message")

    input_language = detect(user_input)

    # Translate input to English if it's in Punjabi
    if input_language == "pa":
        user_input = translate_text(user_input, "English")

    body = {
        "dataSources": [
            {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": COGNITIVE_SEARCH_ENDPOINT,
                    "key": COGNITIVE_SEARCH_KEY,
                    "indexName": COGNITIVE_SEARCH_INDEX_NAME,
                    "semanticConfiguration": None,
                    "queryType": "simple",
                    "fieldsMapping": {
                        "contentFieldsSeparator": "\n",
                        "contentFields": ["content"],
                        "filepathField": None,
                        "titleField": None,
                        "urlField": "url",
                        "vectorFields": [],
                    },
                    "inScope": True,
                    "roleInformation": "You are an AI assistant that helps people find information. answer only if there is relevent documents is there. Do not make up any information and answer strictly from the document. Do not provide full forms to any abbreviations which are not present in the document.",
                    "filter": None,
                    "embeddingEndpoint": None,
                    "embeddingKey": None,
                },
            }
        ],
        "messages": [{"role": "user", "content": user_input}],
        "deployment": OPENAI_ENGINE,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 800,
        "stop": None,
        "stream": False,
    }

    response = requests.post(url, headers=headers, json=body)

    json_response = response.json()

    message = json_response["choices"][0]["messages"][1]["content"]
    

    if input_language == "pa":
        message = translate_text(message, "Punjabi")


    tool_message_content = json_response["choices"][0]["messages"][0]["content"]

    # Converting the content string to a dictionary

    tool_message_content_dict = json.loads(tool_message_content)

    # Extracting the 'citations' field if present
    url2 = ""
    if "citations" in tool_message_content_dict:
        citations = tool_message_content_dict["citations"]
        

        # Extracting the URL from the first citation if present

        if citations:
            first_citation = citations[0]

            if "url" in first_citation:
                url2 = first_citation["url"]

                # print(url2)

            else:
                print("No URL found in the first citation")

        else:
            print("No citations found")
    else:
        print("No 'citations' field found in the tool message content")

    # print(message)
    url2 = url2.replace("/trainingdocuments/", "/originaldocuments/") #  change citiation url to original documents url 

    return jsonify({"assistant_content": message + " " +  url2})
    

if __name__ == "__main__":
    app.run()
