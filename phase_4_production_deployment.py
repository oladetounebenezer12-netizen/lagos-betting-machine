from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
from slowapi import Limiter
from slowapi.util import get_remote_address

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setting up FastAPI
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting
limiter = Limiter(
    key_func=get_remote_address
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Request to {request.url} completed in {process_time} seconds.")
    return response

@app.get("/api/resource")
@limiter.limit("10/minute")
async def read_resource():
    return {"message": "This is a rate-limited endpoint."}

# Production Deployment Details
# Audit Logging and Model Drift Monitoring to be implemented here

# Main execution block
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)