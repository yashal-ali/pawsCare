
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain.vectorstores import FAISS
from langchain.retrievers.multi_vector import MultiVectorRetriever
import os
import uuid
import base64
from fastapi import FastAPI, Request, Form, Response, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
import json
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


prompt_template = """
You are a veterinarian specializing in providing immediate care and support for pets and livestock. Your expertise in diagnosing and treating animals has been instrumental in addressing the challenges faced by pet owners and cattle farmers, especially during late hours and in rural areas.

Prompt: The inspiration for PawsCare stemmed from a personal experience with my beloved dog, Max. One midnight, Max fell seriously ill, and the anxiety of finding immediate help made me realize how difficult it is to access veterinary care during late hours. After talking to various pet owners, I discovered that many face the same issue, struggling to find immediate veterinary assistance when their pets fall ill at inconvenient times. Additionally, I learned that cattle farmers often face similar challenges, as veterinary hospitals are not readily accessible in rural areas. These insights reinforced the need for a solution that provides instant veterinary advice and support, ensuring that all pet owners and farmers have access to the care they need, no matter the time or location. This led to the creation of PawsCare.

Question: {question}
Answer:
"""

qa_chain = LLMChain(llm=ChatOpenAI(model="gpt-4", openai_api_key = openai_api_key, max_tokens=1024),
                    prompt=PromptTemplate.from_template(prompt_template))

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_answer")
async def get_answer(question: str = Form(...)):
    relevant_docs = db.similarity_search(question)
    context = ""
    relevant_images = []
    for d in relevant_docs:
        if d.metadata['type'] == 'text':
            context += '[text]' + d.metadata['original_content']
        elif d.metadata['type'] == 'table':
            context += '[table]' + d.metadata['original_content']
        elif d.metadata['type'] == 'image':
            context += '[image]' + d.page_content
            relevant_images.append(d.metadata['original_content'])
    result = qa_chain.run({'context': context, 'question': question})
    return JSONResponse({"relevant_images": relevant_images[0], "result": result})
