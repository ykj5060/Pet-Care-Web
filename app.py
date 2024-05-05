from fastapi import FastAPI, Depends, HTTPException, status, Request, Form, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from passlib.context import CryptContext
import uvicorn
import json
from langchain import SagemakerEndpoint
from langchain.prompts import PromptTemplate
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from pymongo import MongoClient
import re

region = "us-east-1"
endpoint_name = "**********************"
ATLAS_CONNECTION_STRING = "mongodb+srv://<USERNAME>:<PASSWORD>@vectordb.jmofzvf.mongodb.net/"
client = MongoClient(ATLAS_CONNECTION_STRING)

# Define collection and index name
db_name = "pet_care"
collection_name = "vector_db"
atlas_collection = client[db_name][collection_name]
vector_search_index = "vector_index"

app = FastAPI()

# Configure templates
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable session management
app.add_middleware(SessionMiddleware, secret_key="your_secret_key")

# Simulated database for users
fake_users_db = []

# User model
class User:
    def __init__(self, username: str, hashed_password: str):
        self.username = username
        self.hashed_password = hashed_password

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Hashed passwords for the sample users (You should hash real passwords before storing them)
hashed_password_user1 = pwd_context.hash("admin@123")  # Replace with the actual hashed password
hashed_password_user2 = pwd_context.hash("user@123")   # Replace with the actual hashed password

# Create sample users
fake_users_db.append(User(username="admin", hashed_password=hashed_password_user1))
fake_users_db.append(User(username="user", hashed_password=hashed_password_user2))

# Function to verify password
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Function to get a user by username
def get_user(username: str):
    for user in fake_users_db:
        if user.username == username:
            return user


def build_chain():

    # Sentence transformer
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large",
                                       model_kwargs={'device': 'cpu'})

    # Laod Faiss index
    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        ATLAS_CONNECTION_STRING,
        f"{db_name}.{collection_name}",
        embeddings,
        index_name="vector_index"
    )

    # Custom ContentHandler to handle input and output to the SageMaker Endpoint
    class ContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "do_sample": True,
                    "top_p": 0.6,
                    "temperature": 0.8,
                    "top_k": 100,
                    "max_new_tokens": 1000,
                    "repetition_penalty": 1.03,
                    "stop": ["</s>"]
                }
            }
            input_str = json.dumps(
                payload,
            )
            return input_str.encode("utf-8")

        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            content = response_json[0]["generated_text"]
            return content

    # Langchain chain for invoking SageMaker Endpoint
    llm = SagemakerEndpoint(
        endpoint_name=endpoint_name,
        region_name=region,
        content_handler=ContentHandler(),
        callbacks=[StreamingStdOutCallbackHandler()],
        endpoint_kwargs={"CustomAttributes": "accept_eula=true"},
    )
    prompt_template = """
    You are Dr. Whisker, a cat-like chatbot renowned for being cute, vivid, humorous, caring, gentle, earnest, responsible, supportive, and always respectful and honest. You are a specialist in pet care. Your responses are designed to be beneficial, prioritizing safety and ensuring no harmful, unethical, racist, sexist, toxic, dangerous, or illegal content is included. You aim to maintain a socially unbiased and positive stance in all interactions.

    Text: {context}
    Question: {question}

    As a pet care expert, provide a detailed and accurate response based on the text and question provided. If the text lacks the necessary information to formulate an answer, clearly explain the reason and suggest alternative ways the user might obtain the correct information. Ensure your answers are comprehensible and specifically tailored to help pet owners effectively. Remember to maintain Dr. Whisker's personality in your responses.
    Helpful Answer:
    """


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = { "prompt" : PROMPT } 
    # Langchain chain for Conversation
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_search.as_retriever(search_type="similarity",search_kwargs={"k": 5},),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    return qa


def run_chain(chain, prompt: str):  
    return chain({"query": prompt})

chain = build_chain()

def process_answer(answer):
    answer_list = answer.split("\n")
    answer = "\n".join(answer_list)
    answer1 = answer.split("Question:")[0]
    print("Answer1: ", answer1)
    answer2 = answer.split("Helpful Answer:")[1]
    print("Answer2: ", answer2)
    # # final_answer = answer1 + '\n' + answer2

    return answer2


# Login route
@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = get_user(username)
    if user is None or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # For simplicity, store user information in session and redirect to a protected page
    request.session['user'] = user.username
    response_data = jsonable_encoder(json.dumps({"msg": "Success",}))
    res = Response(response_data)
    return res

# Logout route
@app.get("/logout")
async def logout(request: Request):
    # Remove user information from session
    request.session.pop('user', None)
    return RedirectResponse(url="/")

@app.get("/index")
async def chat(request: Request):
    # Check if user is logged in
    user = request.session.get('user')
    if user is None:
        # Redirect to login if not logged in
        return RedirectResponse(url="/")
    return templates.TemplateResponse("index.html", {"request": request, "user": user})


# Chat route (protected)
@app.get("/chat")
async def chat(request: Request):
    # Check if user is logged in
    user = request.session.get('user')
    if user is None:
        # Redirect to login if not logged in
        return RedirectResponse(url="/")
    return templates.TemplateResponse("chat.html", {"request": request, "user": user})

# Root route
@app.get("/")
async def read_root(request: Request):
    # Check if user is logged in
    user = request.session.get('user')
    if user is None:
        # Display login page if not logged in
        return templates.TemplateResponse("login.html", {"request": request})
    else:
        # Redirect to chat if logged in
        return RedirectResponse(url="/chat")

@app.post("/chat_response")
async def chat_resonse(request: Request, prompt: str = Form(...)):
    result = run_chain(chain=chain, prompt=prompt)
    answer = process_answer(result['result'])
    source_documents = result['source_documents']
    source_documents_list = []
    page_number_list = []
    for doc in source_documents:
        source_doc = doc.metadata['source']
        page_number = doc.metadata['page']
        if source_doc not in source_documents_list:
            source_documents_list.append(source_doc)
            page_number_list.append(page_number)

    response_data = jsonable_encoder(json.dumps({"answer": answer, "source_documents_list": source_documents_list, "page_number_list": page_number_list}))
    res = Response(response_data )
    return res

if __name__ == "__main__":
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=True)