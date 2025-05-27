from fastapi import FastAPI
from pydantic import BaseModel
from ragengine import get_response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


# Add the CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow these origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, OPTIONS, etc)
    allow_headers=["*"],  # Allow all headers
)


# Define the request body model
class QueryRequest(BaseModel):
    query: str


# Define the response model
class QueryResponse(BaseModel):
    response: str
    metadata: list


# Function to process the query
def process_query(query: str) -> str:
    # Implement your processing logic here
    file_directory = "./data"
    response = get_response(file_directory, user_query=query)
    meta_data = []
    for src in response.source_nodes:
        meta_data.append(src.node.metadata)

    return response.response, meta_data


# Define the POST endpoint
@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    result, metadata = process_query(request.query)
    return QueryResponse(response=result, metadata=metadata)
