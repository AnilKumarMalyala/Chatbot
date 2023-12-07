from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
from flask import Flask, request, jsonify

app = FastAPI()

class Request(BaseModel):
    text: str


@app.post("/")
async def home(request: Request):
    try:
        response_text =  request.text + "Hi there from back end"
        if response_text:
            response_json = {
                "fulfillmentText": response_text,
                "fulfillmentMessages": [{"text": {"text": [response_text]}},],
                "payload": {"responseText": response_text}
            }
            
            
            res = {"fulfillment_response": {"messages": [{"text": {"text": [response_text]}}]}}
            return res


        raise Exception("No fulfillment text found in request")

    except Exception as e:
        print(str(e))
        return JSONResponse(content={"error": "Invalid request format"}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
