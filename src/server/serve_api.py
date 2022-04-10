#!/usr/bin/env python3

from typing import Optional
import threading
import queue
import time
from loguru import logger

import pydantic
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        )

request_queue = queue.Queue()
response_queue = queue.Queue()

def worker():
    from model import inference
    import jax
    model = inference.Inference(path="../model_slim/step_88001/")
    with jax.experimental.maps.mesh(inference._devices, ("dp", "mp")):
        while True:
            start_time = time.time()
            request = request_queue.get()
            logger.info(f"getting request took {time.time() - start_time}")
            start_time = time.time()
            response = model.generate(
                    prompt=request.prompt,
                    length=request.length,
                    top_p=request.top_p,
                    temperature=request.temperature,
                    )
            logger.info(f"generate took {time.time() - start_time}")
            start_time = time.time()
            response_queue.put(response)
            logger.info(f"putting response took {time.time() - start_time}")

@app.on_event("startup")
def startup():
    threading.Thread(
            target=worker,
            daemon=True,
            ).start()

@app.get("/")
async def main():
    return {"response": "Hello, world!"}

class CompleteRequest(pydantic.BaseModel):
    prompt: str
    length: Optional[int] = 128
    top_p: Optional[float] = 1.0
    temperature: Optional[float] = 1.0

@app.post("/complete")
def complete(request: CompleteRequest):
    request_queue.put(request)
    response = response_queue.get()
    return {"response": response}
