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

class Settings(pydantic.BaseSettings):
    queue_size: int = 1024
    log_file: str = "logs/serve_api.log"

settings = Settings()

request_queue = queue.Queue(maxsize=settings.queue_size)

def worker():
    from model import inference
    import jax
    model = inference.Inference(path="../model_slim/step_88001/")
    with jax.experimental.maps.mesh(inference._devices, ("dp", "mp")):
        while True:
            try:
                start_time = time.time()
                (request, response_queue) = request_queue.get()
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
                with open(settings.log_file, "a") as f:
                    f.write(f"##### {time.time()} #####\n")
                    f.write(f"{request.prompt}\n")
                    f.write("#####\n")
                    f.write(f"{response}\n\n")
                logger.info(f"writing log took {time.time() - start_time}")
                start_time = time.time()
                response_queue.put(response)
                logger.info(f"putting response took {time.time() - start_time}")
            except KeyboardInterrupt:
                logger.info(f"Got KeyboardInterrupt... quitting!")
                raise
            except Exception:
                logger.exception(f"Got exception, will continue")
                response_queue.put("")

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
    prompt: pydantic.constr(min_length=1, max_length=1024)
    length: pydantic.conint(ge=1, le=1024) = 128
    top_p: pydantic.confloat(ge=0.0, le=1.0) = 1.0
    temperature: pydantic.confloat(ge=0.0) = 1.0

@app.post("/complete")
def complete(request: CompleteRequest):
    logger.info(f"Received request. Queue size is {request_queue.qsize()}")
    if request_queue.full():
        logger.warning("Request queue full.")
        raise ValueError("Request queue full.")
    response_queue = queue.Queue()
    request_queue.put((request, response_queue))
    response = response_queue.get()
    return {"response": response}
