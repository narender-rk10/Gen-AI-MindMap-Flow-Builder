from fastapi import FastAPI, HTTPException, UploadFile, status, File, Depends, Form
from pymongo import MongoClient
from bson import ObjectId
import boto3
import pathlib
from typing import List
from Models.model import CSVNodeQueryRequest, CSVNodeQueryResponse, Flow
from Models.model import PDFNodeQueryRequest
from Models.model import PDFNodeQueryResponse
from Models.model import XLSXNodeQueryRequest
from Models.model import XLSXNodeQueryResponse
from Models.model import ComponentFollowUpQueryRequest
from Models.model import ComponentFollowUpQueryResponse
from Models.model import MultipleQuestionAnswerQueryRequest
from Models.model import MultipleQuestionAnswerQueryResponse
from Models.model import FlowSummarizeRequest
from Models.model import FlowSummarizeResponse
from pymongo.mongo_client import MongoClient
from botocore.exceptions import ClientError
from hashlib import sha256
import time
import pandas as pd
import chromadb
from uuid import uuid4
import datetime
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import List
import os
import json
import re
from dotenv import load_dotenv
import os, re, json, traceback
import boto3
from typing import Optional, Tuple

load_dotenv()

mongo_db_url = os.getenv("mongo_db_url")
aws_access_key_id_str = os.getenv("aws_access_key_id")
aws_secret_access_key_str = os.getenv("aws_secret_access_key")
bucket_name = os.getenv("bucket_name")

AWS_REGION = os.getenv("AWS_BEDROCK_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_NOVA_MODEL_ID", "amazon.nova-lite-v1:0")
S3_BUCKET = os.getenv("bucket_name")
MAKE_PUBLIC = True  

MB = 1024 * 1024
MAX_ANY_PAYLOAD = 25 * MB       

TEXT_DOC_EXTS = {"csv", "xls", "xlsx"}
MEDIA_DOC_EXTS = {"pdf"}  

CONTENT_TYPES = {
    "pdf":  "application/pdf",
    "csv":  "text/csv",
    "doc":  "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xls":  "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "html": "text/html",
    "txt":  "text/plain",
    "md":   "text/markdown",
    "png":  "image/png",
    "jpg":  "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif":  "image/gif",
    "bmp":  "image/bmp",
    "tiff": "image/tiff",
    "mp4":  "video/mp4",
    "mov":  "video/quicktime",
    "mkv":  "video/x-matroska",
    "webm": "video/webm",
    "flv":  "video/x-flv",
    "mpeg": "video/mpeg",
    "mpg":  "video/mpeg",
    "wmv":  "video/x-ms-wmv",
    "3gp":  "video/3gpp",
}

ALLOWED_EXTS = set(CONTENT_TYPES.keys())


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


client = MongoClient(mongo_db_url)

try:
    client.admin.command("ping")
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client["MindMap"]
flow_collection = db["flows"]
component_collection = db["components"]
node_collection = db["nodes"]

s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id_str,
    aws_secret_access_key=aws_secret_access_key_str,
    region_name="ap-south-1",
)


s3_client = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

def _https_to_s3_uri(url: str) -> str | None:
    """
    Convert https://<bucket>.s3[.<region>].amazonaws.com/<key> -> s3://<bucket>/<key>
    Returns None if it doesn't look like an S3 URL.
    """
    if not url:
        return None
    m = re.match(r"https?://([^.]+)\.s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com/(.+)", url)
    if not m:
        return None
    bucket, key = m.group(1), m.group(2)
    return f"s3://{bucket}/{key}"

CITATION_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "quote":       {"type": "string"},   # short text the answer relied on
        "modality":    {"type": "string", "enum": ["text","image","video","audio","web","table","other"]},
        "s3_uri":      {"type": "string"},   # source in S3 if known
        "url":         {"type": "string"},   # public URL (web/YT/etc.) if known

        # Text/PDF/Web
        "page":        {"type": "integer"},  # PDF page (1-based)
        "section":     {"type": "string"},

        # CSV/XLSX/Table
        "rowStart":    {"type": "integer"},
        "rowEnd":      {"type": "integer"},

        # Image
        "bbox": {
            "type": "object",
            "properties": {"x":{"type":"number"}, "y":{"type":"number"}, "w":{"type":"number"}, "h":{"type":"number"}},
            "additionalProperties": False
        },

        # Video/Audio
        "startTimeSec":{"type": "number"},
        "endTimeSec":  {"type": "number"},
        "frameIndex":  {"type": "integer"},

        "confidence":  {"type": "number"}    # 0..1
    },
    "required": ["quote"],
    "additionalProperties": False
}

DOC_QA_TOOL_CONFIG = {
    "tools": [{
        "toolSpec": {
            "name": "AnswerFormat",
            "description": "Structured answer for document Q&A",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "summ":  {"type": "string"},
                        "df":    {"type": "array", "items": {"type": "object"}},
                        "graph":     {"type": "string"},
                        "citations": {"type": "array", "items": CITATION_ITEM_SCHEMA}
                    },
                    "required": ["summ", "df", "graph", "citations"],
                    "additionalProperties": False
                }
            }
        }
    }]
}

ALLOWED_DOC_EXTS = {"pdf","csv","xls","xlsx"}
MAX_NOVA_BYTES   = 25 * 1024 * 1024  # 25MB

def _sanitize_doc_name_and_ext(key: str) -> Tuple[str, str]:
    """
    For Bedrock document:
      - 'name' must NOT contain '.' and only allow [A-Za-z0-9 space - ( ) [ ]]
      - 'format' is the lowercase file extension (no dot)
    Returns (safe_name, ext)
    """
    fname = key.split("/")[-1] if "/" in key else key
    if "." in fname:
        stem, ext = fname.rsplit(".", 1)
        ext = ext.lower()
    else:
        stem, ext = fname, ""
    stem = stem.strip()
    stem = re.sub(r"[^A-Za-z0-9\-\ \(\)\[\]]+", " ", stem)  # remove illegal chars (including dots/underscores)
    stem = re.sub(r"\s{2,}", " ", stem)                      # collapse multiple spaces
    if not stem:
        stem = "document"
    return stem, ext

def _https_to_s3_uri(url: str) -> str | None:
    """
    Convert https://<bucket>.s3[.<region>].amazonaws.com/<key> -> s3://<bucket>/<key>
    Returns None if it doesn't look like an S3 URL.
    """
    if not url:
        return None
    m = re.match(r"https?://([^.]+)\.s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com/(.+)", url)
    if not m:
        return None
    bucket, key = m.group(1), m.group(2)
    return f"s3://{bucket}/{key}"

def _s3_bytes_block_from_uri(s3_uri: str) -> Optional[dict]:
    """
    Build a Bedrock 'document' content block using S3 bytes.
    Returns the content block or None if invalid/too large/unsupported.
    """
    if not s3_uri or not s3_uri.startswith("s3://"):
        print("⚠️ Missing or invalid s3_uri")
        return None

    without = s3_uri[5:]
    parts = without.split("/", 1)
    if len(parts) != 2:
        print(f"⚠️ Malformed s3_uri: {s3_uri}")
        return None

    bucket, key = parts[0], parts[1]

    head = s3_client.head_object(Bucket=bucket, Key=key)
    size = head.get("ContentLength", 0)
    if size > MAX_NOVA_BYTES:
        print(f"⚠️ Skipping attach: {key} is {size} bytes (>25MB)")
        return None

    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()

    safe_name, ext = _sanitize_doc_name_and_ext(key)
    if ext not in ALLOWED_DOC_EXTS:
        print(f"⚠️ Unsupported ext '.{ext}' for Bedrock document attach: {key}")
        return None

    block = {
        "document": {
            "name": safe_name,
            "format": ext,
            "source": {"bytes": data}
        }
    }
    print(f"Attaching document → name='{safe_name}', format='{ext}', bytes={len(data)}")
    return block

def bedrock_nova_doc_qa(
    s3_uri: str,
    query: str,
    persona_instructions: str = "",
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 1800
) -> dict:
    """
    Generic Bedrock Nova Lite doc QA that returns a dict {summ, df, graph}.
    If attach fails (too big/unsupported), still prompts without bytes.
    Never raises for content; returns empty fields on error.
    """
    
    CITATION_GUIDE = (
    "CITATION RULES:\n"
    "- Always include a non-empty citations[] array when you answer.\n"
    "- Each citation must include a short 'quote' you relied on.\n"
    "- If the source is a PDF, include 'page' (1-based). If a table (CSV/XLSX), include 'rowStart' and 'rowEnd' for the rows used.\n"
    "- If a web/markdown doc, include a 'section' (heading or anchor).\n"
    "- If an image, include 'bbox' with normalized coordinates {x,y,w,h} in [0..1].\n"
    "- If a video/audio, include 'startTimeSec' and 'endTimeSec' (or 'frameIndex').\n"
    "- Also include 's3_uri' when available; add 'url' if a public URL was provided.\n"
    )


    sys_text = (
        "You answer ONLY based on the provided document. "
        "Return a single AnswerFormat tool call with fields: summ (string), df (array of JSON rows), graph (string), citations (array).\n" + CITATION_GUIDE + "\n"
        "If no relevant answer is possible, return empty answer with df=[] and graph=\"\". "
        "If tabular data is present, include it in df (array of objects). "
        "If a chart makes sense, return a Plotly JSON string in graph. "
        "The app uses a dark background for graphs."
    )
    user_text = f"QUESTION: {query}\nPERSONA: {persona_instructions or 'Default'}"

    content_blocks = []
    doc_block = _s3_bytes_block_from_uri(s3_uri)
    if doc_block:
        content_blocks.append(doc_block)
    else:
        # provide filename hint if we couldn't attach bytes
        hint = (s3_uri.split('/')[-1] if s3_uri else 'unknown')
        content_blocks.append({"text": f"(Reference file: {hint}. Unable to attach bytes.)"})

    messages = [
        {"role": "user", "content": [{"text": sys_text}]},
        {"role": "user", "content": content_blocks + [{"text": user_text}]}
    ]

    # Default empty output
    result = {"summ": "", "df": [], "graph": ""}

    try:
        resp = bedrock.converse(
            modelId="amazon.nova-lite-v1:0",
            messages=messages,
            toolConfig=DOC_QA_TOOL_CONFIG,
            inferenceConfig={
                "temperature": float(temperature),
                "topP": float(top_p),
                "maxTokens": int(max_tokens)
            },
        )
        print(resp)
        blocks = resp["output"]["message"]["content"]
        tool_blocks = [b for b in blocks if "toolUse" in b]
        if tool_blocks:
            tu = next(
                (b["toolUse"] for b in tool_blocks if b["toolUse"].get("name") == "AnswerFormat"),
                tool_blocks[0]["toolUse"]
            )
            payload = tu.get("input", {}) or {}
            # Ensure keys exist & correct types
            result["summ"]  = str(payload.get("summ", "")).strip()
            df_val          = payload.get("df", [])
            result["df"]    = df_val if isinstance(df_val, list) else []
            result["graph"] = str(payload.get("graph", "")).strip()
        else:
            # Fallback: try parse any JSON-looking text
            txts = [b.get("text") for b in blocks if "text" in b and b.get("text")]
            if txts:
                raw = txts[0].strip().strip("```json").strip("```").strip()
                try:
                    payload = json.loads(raw)
                    result["summ"]  = str(payload.get("summ",""))
                    result["df"]    = payload.get("df", []) if isinstance(payload.get("df", []), list) else []
                    result["graph"] = str(payload.get("graph",""))
                except Exception:
                    pass
        return result

    except Exception as e:
        print("❌ Bedrock converse FAILED:", repr(e))
            
    

def _enforce_size_rules(ext: str, size_bytes: int):
    if size_bytes > MAX_ANY_PAYLOAD:
        raise ValueError(f"File too large ({size_bytes/MB:.2f} MB). Max allowed is 25 MB.")

def _upload_to_s3(flow_id: str, filename: str, file_bytes: bytes, content_type: str) -> str:
    # We avoid spaces in the "folder" for cleaner URLs.
    key = f"{flow_id}/auto-mind-map/{filename}"
    extra_args = {"ContentType": content_type}
    if MAKE_PUBLIC:
        extra_args["ACL"] = "public-read"
    s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=file_bytes, **extra_args)
    return f"s3://{S3_BUCKET}/{key}"


def sanitize_filename(filename: str) -> str:
    # Separate extension
    match = re.match(r"^(.*?)(\.[^.]+)?$", filename)
    if not match:
        return "file"  # fallback
    name, ext = match.groups()

    # Keep only allowed characters: alnum, space, hyphen, (), []
    name = re.sub(r"[^a-zA-Z0-9\s\-\(\)\[\]]", "-", name)

    # Collapse multiple whitespace into single space
    name = re.sub(r"\s+", " ", name).strip()

    # Ensure it’s not empty
    if not name:
        name = "file"

    # Reattach extension (ext includes the ".")
    return f"{name}{ext}" if ext else name

def sanitize_for_bedrock_name(filename: str) -> str:
    # Only keep the stem (no extension)
    stem = pathlib.Path(filename).stem
    # Replace disallowed chars with hyphen
    clean = re.sub(r"[^a-zA-Z0-9\-\(\)\[\]\s]", "-", stem)
    # Collapse whitespace
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        clean = "file"
    return clean  # no extension

def validate_dataframe(df):
    try:
        if not isinstance(df, list) and not all(isinstance(item, dict) for item in df):
            return []
        else:
            return df

    except ValueError:
        return []

def fetch_question_answer_from_node_collection(parent_id: str, flow_id: str):
    try:
        record = node_collection.find_one(
            {
                "_id": ObjectId(parent_id),
                "flow_id": ObjectId(flow_id),
                "is_delete": "false",
            }
        )

        print("Fetched record:", record)

        if not record:
            return None, None

        print("Question value:", record.get("question"))

        question = record.get("question", None)
        answer = None

        if record["type"] == "pdf":
            answer = "Answer: " + str(record.get("summ", "Answer not found."))
            df_list = record.get("df", [])
            df_string = " | ".join([str(item) for item in df_list])

            answer += " DataFrame: " + df_string


        elif record["type"] == "csv":
            answer = "Answer: " + str(record.get("summ", "Answer not found."))
            df_list = record.get("df", [])
            df_string = " | ".join([str(item) for item in df_list])
            
            answer += " DataFrame: " + df_string
            
        elif record["type"] == "xlsx":
            answer = "Answer: " + str(record.get("summ", "Answer not found."))
            df_list = record.get("df", [])
            df_string = " | ".join([str(item) for item in df_list])
            
            answer += " DataFrame: " + df_string

        elif record["type"] == "MultipleQA":
            answer = "Answer: " + str(record.get("summ", "Answer not found."))
            df_list = record.get("df", [])
            df_string = " | ".join([str(item) for item in df_list])
            
            answer += " DataFrame: " + df_string

        print("Answer:", answer)

        return question, answer
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")
        return None, None


@app.post("/create-flow/")
def create_flow(flow: dict):
    try:
        flow_data = {"flow_name": flow.get("flow_name"), "flow_json": "", "summary": "", "flow_type": flow.get("flow_type")}
        flow_id = flow_collection.insert_one(flow_data).inserted_id
        flow_type = flow.get("flow_type")
        return {"flow_id": str(flow_id), "flow_type": str(flow_type)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating flow: {str(e)}")


@app.delete("/delete-flow/{flow_id}")
def delete_flow(flow_id: str):
    try:
        flow_object_id = ObjectId(flow_id)
        components = component_collection.find({"flow_id": flow_object_id})
        for component in components:
            component_id = component["_id"]
            node_collection.delete_many({"component_id": component_id})
        component_collection.delete_many({"flow_id": flow_object_id})

        flow_collection.delete_one({"_id": flow_object_id})

        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting flow: {str(e)}")


@app.get("/flows/", response_model=List[Flow])
def list_flows():
    flows = flow_collection.find()
    return [
        {
            "flow_id": str(flow["_id"]),
            "flow_name": flow["flow_name"],
            "flow_json": flow["flow_json"],
            "summary": flow["summary"],
            "flow_type": flow["flow_type"]
        }
        for flow in flows
    ]


@app.put("/flow-update/")
def update_flow(update_data: Flow):
    try:
        print(update_data)
        result = flow_collection.update_one(
            {"_id": ObjectId(update_data.flow_id)},
            {
                "$set": {
                    "flow_name": update_data.flow_name,
                    "flow_json": update_data.flow_json,
                    "flow_type": update_data.flow_type,
                    "summary": update_data.summary,
                }
            },
        )
        print(result)

        if result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Flow not found"
            )

        return {
            "flow_id": str(update_data.flow_id),
            "message": "Flow updated successfully",
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {str(e)}",
        )

def _choose_prefix(flow_id: str, flow_type: str) -> str:
    ft = (flow_type or "").lower()
    if "manual" in ft:
        return f"{flow_id}/manual-mind-map/"
    return f"{flow_id}/auto-mind-map/"

def get_summary_from_bedrock(file: UploadFile, flow_id: str, flow_type: str):
    try:
        file.file.seek(0)
        file_bytes = file.file.read()
        if not file_bytes:
            raise ValueError("The uploaded file is actually empty!")

        if len(file_bytes) > MAX_ANY_PAYLOAD:
            raise ValueError("File too large. Max size is 25 MB.")

        safe_name = sanitize_filename(file.filename)
        ext = (safe_name.split(".")[-1] if "." in safe_name else "").lower()
        if ext not in ALLOWED_EXTS:
            raise ValueError(f"Unsupported file type '.{ext}'.")

        content_type = CONTENT_TYPES.get(ext, "application/octet-stream")

        prefix = _choose_prefix(flow_id, flow_type)
        s3_key = f"{prefix}{safe_name}"

        put_kwargs = {
            "Bucket": S3_BUCKET,
            "Key": s3_key,
            "Body": file_bytes,
            "ContentType": content_type,
        }
        
        put_kwargs["ACL"] = "public-read"
        print(s3_client, s3_key, put_kwargs)
        s3_client.put_object(**put_kwargs)


        s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
        public_url = (
            f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"
        )

        doc_filter = {"flow_id": ObjectId(flow_id), "name": safe_name}
        doc_update = {
            "$set": {
                "flow_id": ObjectId(flow_id),
                "name": safe_name,
                "size": len(file_bytes),
                "type": ext,
                "processing_type": "bedrock",   
                "file_id": None,
                "assistant_id": None,
                "vector_store_id": None,
                "summary": None,
                "s3_uri": s3_uri,
                "public_url": public_url,
                "flow_type": flow_type,
            }
        }

        existing = component_collection.find_one(doc_filter)
        if existing:
            component_collection.update_one({"_id": existing["_id"]}, doc_update)
            component_id = existing["_id"]
        else:
            component_id = component_collection.insert_one(
                {**doc_update["$set"]}
            ).inserted_id

        return {
            "component_id": str(component_id),
            "type": ext,
            "flow_type": flow_type
        }

    except Exception as e:
        import traceback
        print("❌ Error in get_summary_from_bedrock:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/component-create-pdf")
def create_pdf_component(
    file: UploadFile, flow_id: str = Form(...), processing_type: str = Form(...)
):
    flow = flow_collection.find_one({"_id": ObjectId(flow_id)})
    if file.filename.endswith(".pdf"):
        if flow["flow_type"] == 'manual':
            return get_summary_from_bedrock(file, flow_id=flow_id, flow_type='manual')
        else:
            traceback.print_exc()
            return HTTPException(status_code=404, detail="Exceeded limit for AWS BedRock.")
    else:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")



@app.post("/component-create-xlsx")
def create_xlsx_component(file: UploadFile, flow_id: str = Form(...)):
    flow = flow_collection.find_one({"_id": ObjectId(flow_id)})
    if file.filename.endswith(".xls") or file.filename.endswith(".xlsx"):
        if flow["flow_type"] == 'manual':
            return get_summary_from_bedrock(file, flow_id=flow_id, flow_type=flow["flow_type"])
        else:
            traceback.print_exc()
            return HTTPException(status_code=404, detail="Some Error occurred in XLSX file processing.")
    else:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail="Only XLSX or XLS files are allowed.")



@app.post("/component-create-csv")
def create_csv_component(
    file: UploadFile = File(...), flow_id: str = Form(...)
):
    flow = flow_collection.find_one({"_id": ObjectId(flow_id)})
    if file.filename.endswith(".csv"):
        if  flow["flow_type"] == 'manual':
            return get_summary_from_bedrock(file, flow_id=flow_id, flow_type=flow["flow_type"])
        else:
            traceback.print_exc()
            return HTTPException(status_code=404, detail="Exceeded Page limit for Bedrock.")
    else:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

from io import BytesIO


@app.post("/pdf-component-qa", response_model=List[PDFNodeQueryResponse])
def PDF_QA(request: PDFNodeQueryRequest):
    try:
        record = component_collection.find_one(
            {
                "flow_id": ObjectId(request.flow_id),
                "_id": ObjectId(request.component_id),
                "type": "pdf",
            }
        )
        if not record:
            raise HTTPException(status_code=404, detail="Component not found")
        if "s3_uri" not in record or not record["s3_uri"]:
            raise HTTPException(status_code=400, detail="Component missing s3_uri")

        persona_instructions = (
            record.get("instructions")
        )

        qa = bedrock_nova_doc_qa(
            s3_uri=record["s3_uri"],
            query=request.query,
            persona_instructions=persona_instructions,
            temperature=(record.get("temperature") if record.get("temperature") is not None else 0.2),
            top_p=(record.get("top_p") if record.get("top_p") is not None else 0.9),
            max_tokens=7800
        )

        summ  = qa.get("summ", "")
        df    = validate_dataframe(qa.get("df", []))
        graph = qa.get("graph", "")

        node_data = {
            "_id": ObjectId(request.node_id),
            "flow_id": ObjectId(request.flow_id),
            "component_id": ObjectId(request.component_id),
            "question": request.query,
            "summ": summ,
            "df": df,
            "graph": graph,
            "type": "pdf",
            "is_delete": "false",
            "timestamp": datetime.datetime.utcnow(),
        }
        node_id_response = node_collection.insert_one(node_data)
        print("Inserted node:", node_id_response)

        question_entries = [
            PDFNodeQueryResponse(
                data={
                    "question": request.query,
                    "summ": summ,
                    "df": df,
                    "graph": graph,
                    "flow_id": request.flow_id,
                    "component_id": request.component_id,
                    "component_type": "pdf",
                },
                id=request.node_id,
                type="PDFNode",
            )
        ]

        if request.request_type == "question":
            empty_question_entry = PDFNodeQueryResponse(
                id=str(ObjectId()),
                data={
                    "question": "",
                    "flow_id": request.flow_id,
                    "component_id": request.component_id,
                    "component_type": "pdf",
                },
                type="question",
            )
            question_entries.append(empty_question_entry)

        return question_entries

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        print(f"Error in /pdf-component-qa endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    

@app.post("/xlsx-component-qa", response_model=List[XLSXNodeQueryResponse])
def XLSX_QA(request: XLSXNodeQueryRequest):
    try:
        record = component_collection.find_one(
            {
                "flow_id": ObjectId(request.flow_id),
                "_id": ObjectId(request.component_id),
                "type": {"$in": ["xlsx", "xls"]},
            }
        )
        if not record:
            raise HTTPException(status_code=404, detail="Component not found")
        if "s3_uri" not in record or not record["s3_uri"]:
            raise HTTPException(status_code=400, detail="Component missing s3_uri")

        persona_instructions = record.get("instructions")

        qa = bedrock_nova_doc_qa(
            s3_uri=record["s3_uri"],
            query=request.query,
            persona_instructions=persona_instructions,
            temperature=(record.get("temperature") if record.get("temperature") is not None else 0.2),
            top_p=(record.get("top_p") if record.get("top_p") is not None else 0.9),
            max_tokens=7800,
        )

        summ  = qa.get("summ", "")
        df    = validate_dataframe(qa.get("df", []))
        graph = qa.get("graph", "")

        node_data = {
            "_id": ObjectId(request.node_id),
            "flow_id": ObjectId(request.flow_id),
            "component_id": ObjectId(request.component_id),
            "question": request.query,
            "summ": summ,
            "df": df,
            "graph": graph,
            "type": "xlsx",
            "is_delete": "false",
            "timestamp": datetime.datetime.utcnow(),
        }
        node_collection.insert_one(node_data)

        entries: List[XLSXNodeQueryResponse] = [
            XLSXNodeQueryResponse(
                data={
                    "question": request.query,
                    "summ": summ,
                    "df": df,
                    "graph": graph,
                    "flow_id": request.flow_id,
                    "component_id": request.component_id,
                    "component_type": "xlsx",
                },
                id=request.node_id,
                type="XLSXNode",
            )
        ]

        if request.request_type == "question":
            entries.append(
                XLSXNodeQueryResponse(
                    id=str(ObjectId()),
                    data={
                        "question": "",
                        "flow_id": request.flow_id,
                        "component_id": request.component_id,
                        "component_type": "xlsx",
                    },
                    type="question",
                )
            )

        return entries

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in /xlsx-component-qa endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/csv-component-qa", response_model=List[CSVNodeQueryResponse])
def CSV_QA(request: CSVNodeQueryRequest):
    try:
        record = component_collection.find_one(
            {
                "flow_id": ObjectId(request.flow_id),
                "_id": ObjectId(request.component_id),
                "type": "csv",
            }
        )
        if not record:
            raise HTTPException(status_code=404, detail="Component not found")
        if "s3_uri" not in record or not record["s3_uri"]:
            raise HTTPException(status_code=400, detail="Component missing s3_uri")

        persona_instructions = record.get("instructions") or ""

        qa = bedrock_nova_doc_qa(
            s3_uri=record["s3_uri"],
            query=request.query,
            persona_instructions=persona_instructions,
            temperature=(record.get("temperature") if record.get("temperature") is not None else 0.2),
            top_p=(record.get("top_p") if record.get("top_p") is not None else 0.9),
            max_tokens=7800,
        )

        summ  = qa.get("summ", "")
        df    = validate_dataframe(qa.get("df", []))
        graph = qa.get("graph", "")

        result_document = {
            "_id": ObjectId(request.node_id),
            "question": request.query,
            "df": df,
            "summ": summ,
            "graph": graph,
            "flow_id": ObjectId(request.flow_id),
            "component_id": ObjectId(request.component_id),
            "type": "csv",
            "is_delete": "false",
            "created_at": datetime.datetime.utcnow(),
        }
        node_collection.insert_one(result_document)

        question_entries = []

        response_data = {
            "id": str(request.node_id),
            "type": "CSVNode",
            "data": {
                "question": request.query,
                "df": df,
                "summ": summ,
                "graph": graph,
                "flow_id": request.flow_id,
                "component_id": request.component_id,
                "component_type": "csv",
            },
        }
        question_entries.append(response_data)

        if request.request_type == "question":
            empty_node = {
                "id": str(ObjectId()),
                "type": "question",
                "data": {
                    "question": "",
                    "flow_id": request.flow_id,
                    "component_id": request.component_id,
                    "component_type": "csv",
                },
            }
            question_entries.append(empty_node)

        return question_entries

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in /csv-component-qa: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.delete("/soft-delete-node/{node_id}", response_model=dict)
def soft_delete_node(node_id: str):
    try:
        result = node_collection.update_one(
            {"_id": ObjectId(node_id)}, {"$set": {"is_delete": "true"}}
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Node not found.")

        return {"message": "Node soft deleted successfully."}

    except Exception as e:
        print(f"Error in /soft-delete-node endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/get-all-flow-details/{flow_id}")
def get_flow_details(flow_id: str):
    try:
        if not ObjectId.is_valid(flow_id):
            raise HTTPException(status_code=400, detail="Invalid flow_id format.")

        flow = flow_collection.find_one({"_id": ObjectId(flow_id)})
        if not flow:
            raise HTTPException(status_code=404, detail="Flow not found.")

        components = list(component_collection.find({"flow_id": ObjectId(flow_id)}))
        if not components:
            raise HTTPException(
                status_code=404, detail="No components found for the given flow_id."
            )

        flow_details = {
            "flow": {
                "flow_id": str(flow["_id"]),
                "flow_name": flow.get("flow_name"),
                "description": flow.get("description"),
                "summary": flow.get("summary"),
            },
            "components": [],
        }

        for component in components:
            component_id = str(component["_id"])
            nodes = list(
                node_collection.find(
                    {
                        "component_id": ObjectId(component_id),
                        "flow_id": ObjectId(flow_id),
                    }
                )
            )

            component_details = {
                "component_id": component_id,
                "name": component.get("name"),
                "file_hash": component.get("file_hash"),
                "size": component.get("size"),
                "s3_path": component.get("s3_path"),
                "nodes": [],
            }

            for node in nodes:
                component_details["nodes"].append(
                    {
                        "node_id": str(node["_id"]),
                        "question": node.get("question"),
                        "answer": node.get("answer"),
                        "timestamp": node.get("timestamp"),
                    }
                )

            flow_details["components"].append(component_details)

        return flow_details

    except Exception as e:
        print(f"Error in /get-all-flow-details/{flow_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


 
@app.post("/components-follow-up-questions", response_model=List[ComponentFollowUpQueryResponse])
def create_follow_up_questions(request: ComponentFollowUpQueryRequest):
    try:
        ctype = request.component_type

        record = component_collection.find_one(
            {"flow_id": ObjectId(request.flow_id), "_id": ObjectId(request.component_id), "type": ctype}
        )
        if not record:
            raise HTTPException(status_code=404, detail="Component not found")

        persona = request.persona_name or record.get("persona_name") or "User"
        instructions = request.instructions or record.get("instructions") or "Generate helpful follow-up questions."
        temperature = request.temperature if request.temperature is not None else 0.2
        top_p = request.top_p if request.top_p is not None else 0.9

        component_collection.update_one(
            {"_id": ObjectId(request.component_id)},
            {"$set": {"instructions": instructions, "persona_name": persona}},
        )

        def _finalize(flow_id, component_id, component_type, questions):
            entries = []
            for q in (questions or []):
                q = (q or "").strip()
                if not q:
                    continue
                entries.append(
                    ComponentFollowUpQueryResponse(
                        id=str(ObjectId()),
                        flow_id=flow_id,
                        data={
                            "question": q,
                            "component_id": component_id,
                            "component_type": component_type,
                        },
                        type="followUp",
                        position={"x": 0, "y": 0},
                    )
                )
            entries.append(
                ComponentFollowUpQueryResponse(
                    id=str(ObjectId()),
                    flow_id=flow_id,
                    position={"x": 0, "y": 0},
                    data={
                        "question": "",
                        "component_id": component_id,
                        "component_type": component_type,
                    },
                    type="question",
                )
            )
            return entries

        # --- helpers ---

        def _https_to_s3_uri(url: str) -> str:
            """
            Convert common HTTPS S3 URLs to s3://bucket/key. Returns "" if not recognized.
            Supports:
              https://<bucket>.s3.amazonaws.com/<key>
              https://s3.<region>.amazonaws.com/<bucket>/<key>
              https://<bucket>.s3.<region>.amazonaws.com/<key>
            """
            try:
                if not url or not url.startswith("http"):
                    return ""
                from urllib.parse import urlparse
                p = urlparse(url)
                host = p.netloc
                path = p.path.lstrip("/")
                # bucket.s3.amazonaws.com/key
                m1 = re.match(r"^([^.]+)\.s3\.amazonaws\.com$", host)
                if m1:
                    return f"s3://{m1.group(1)}/{path}"
                # bucket.s3.region.amazonaws.com/key
                m2 = re.match(r"^([^.]+)\.s3\.[^.]+\.amazonaws\.com$", host)
                if m2:
                    return f"s3://{m2.group(1)}/{path}"
                # s3.region.amazonaws.com/bucket/key
                if host.startswith("s3.") and path:
                    parts = path.split("/", 1)
                    if len(parts) == 2:
                        return f"s3://{parts[0]}/{parts[1]}"
                return ""
            except Exception:
                return ""

        def _s3_doc_or_media_block(s3_uri: str, allow_exts: set):
            """
            Build Bedrock 'document' block from S3 bytes (<=25MB), enforcing allowed extensions.
            Name must exclude file extension and only contain: alnum, spaces, hyphens, (), [].
            """
            if not s3_uri or not s3_uri.startswith("s3://"):
                return None
            without = s3_uri[5:]
            parts = without.split("/", 1)
            if len(parts) != 2:
                return None
            bucket, key = parts[0], parts[1]

            # size guard (25MB)
            head = s3_client.head_object(Bucket=bucket, Key=key)
            size = head.get("ContentLength", 0)
            if size > 25 * 1024 * 1024:
                print(f"⚠️ Skipping attach: {key} is {size} bytes (>25MB)")
                return None

            obj = s3_client.get_object(Bucket=bucket, Key=key)
            data = obj["Body"].read()

            fname = key.split("/")[-1]
            stem, ext = (fname.rsplit(".", 1) + [""])[:2] if "." in fname else (fname, "")
            ext = ext.lower()

            stem = re.sub(r"[^A-Za-z0-9\-\ \(\)\[\]]+", " ", stem).strip()
            stem = re.sub(r"\s{2,}", " ", stem) or "document"

            if ext not in allow_exts:
                print(f"⚠️ Skipping attach: unsupported ext '.{ext}' for {key}")
                return None

            return {
                "document": {
                    "name": stem,       # no extension in name
                    "format": ext,      # put extension here (e.g. pdf, md, xlsx)
                    "source": {"bytes": data}
                }
            }

        tool_config = {
            "tools": [{
                "toolSpec": {
                    "name": "FollowUpQuestions",
                    "description": "Return up to three sharp, non-redundant follow-up questions "
                    "about the provided document/media and persona.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "questions": {"type": "array", "items": {"type": "string"}, "maxItems": 3}
                            },
                            "required": ["questions"],
                            "additionalProperties": False
                        }
                    }
                }
            }]
        }

        DOC_TYPES   = {"pdf","csv","xls","xlsx"}

        # Prefer S3 URI
        s3_uri = record.get("s3_uri") or _https_to_s3_uri(record.get("public_url") or "")


        sys_rules = (
            "You generate concise, insightful follow-up questions based ONLY on the provided document/media and the given persona. "
            "Avoid yes/no and redundancy. If nothing meaningful, return an empty list via FollowUpQuestions."
        )
        persona_block = f"PERSONA: {persona}\nINSTRUCTIONS: {instructions}"

        content_blocks = []
        if ctype in DOC_TYPES:
            block = _s3_doc_or_media_block(
                s3_uri,
                allow_exts={"pdf","txt","md","html","doc","docx","csv","xls","xlsx"}
            )
            if block:
                content_blocks.append(block)
            else:
                hint = (record.get("name") or "document")
                content_blocks.append({"text": f"(Reference file: {hint}. Unable to attach bytes.)"})
        else:
            # Unknown → just return empty input node
            return _finalize(request.flow_id, request.component_id, request.component_type, [])

        content_blocks.append({"text": persona_block})

        messages = [
            {"role": "user", "content": [{"text": sys_rules}]},
            {"role": "user", "content": content_blocks},
        ]

        try:
            resp = bedrock.converse(
                modelId=BEDROCK_MODEL_ID,  # e.g. "amazon.nova-lite-v1:0"
                messages=messages,
                toolConfig=tool_config,
                inferenceConfig={
                    "temperature": float(temperature),
                    "topP": float(top_p),
                    "maxTokens": 1500
                },
            )
            blocks = resp["output"]["message"]["content"]
            tool_blocks = [b for b in blocks if "toolUse" in b]
            if tool_blocks:
                tu = next(
                    (b["toolUse"] for b in tool_blocks if b["toolUse"].get("name") == "FollowUpQuestions"),
                    tool_blocks[0]["toolUse"]
                )
                questions = tu.get("input", {}).get("questions", []) or []
                questions = [str(q).strip() for q in questions if str(q).strip()][:3]
            else:
                txts = [b.get("text") for b in blocks if "text" in b and b.get("text")]
                if txts:
                    raw = txts[0]
                    parts = [p.strip() for p in re.split(r"\|\|\||\r|\n", raw) if p.strip()]
                    questions = parts[:3]
                else:
                    questions = []
        except Exception as e:
            print("❌ Bedrock converse FAILED:", repr(e))
            questions = []

        return _finalize(request.flow_id, request.component_id, request.component_type, questions)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /components-follow-up-questions: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/multiple-qa-summarize", response_model=MultipleQuestionAnswerQueryResponse)
def multiple_qa_summarize(request: MultipleQuestionAnswerQueryRequest):
    try:
        conversation = []
        for parent_id in request.parent_node_ids:
            q, a = fetch_question_answer_from_node_collection(parent_id, request.flow_id)
            if q:
                conversation.append({"role": "user", "content": q})
            if a:
                conversation.append({"role": "assistant", "content": f"Answer:\n{a}"})

        bedrock_history = []
        for turn in conversation:
            role = "assistant" if turn.get("role") == "assistant" else "user"
            text = (turn.get("content") or "").strip()
            if text:
                bedrock_history.append({"role": role, "content": [{"text": text}]})

        TOOL_NAME = "AnswerFormat"
        tool_config = {
            "tools": [{
            "toolSpec": {
            "name": TOOL_NAME,
            "description": "Structured answer for conversation-based summarize/QA",
            "inputSchema": {
            "json": {
            "type": "object",
            "properties": {
            "summ":  {"type": "string"},
            "df":    {"type": "array", "items": {"type": "object"}},
            "graph": {"type": "string"}
            },
            "required": ["summ", "df", "graph"],
            "additionalProperties": False
            }
            }
            }
            }],
            "toolChoice": { "tool": { "name": TOOL_NAME } }
            }

        sys_text = (
            "You are an AI assistant. Read the conversation history and answer the final question.\n"
            f"Respond using a SINGLE {TOOL_NAME} tool call only. Do not include any plain text.\n"
            "Fields:\n"
            " - summ: concise summary/answer\n"
            " - df: array of JSON row objects ([] if none)\n"
            " - graph: Plotly spec as a JSON string (\"\" if none; use dark theme)\n"
        )
        final_q = f"FINAL QUESTION: {request.question}"

        messages = [
            {"role": "user", "content": [{"text": sys_text}]},
            *bedrock_history,
            {"role": "user", "content": [{"text": final_q}]},
        ]

        try:
            resp = bedrock.converse(
                modelId="amazon.nova-lite-v1:0",
                messages=messages,
                toolConfig=tool_config,
                inferenceConfig={
                    "temperature": 0.2,
                    "topP": 0.9,
                    "maxTokens": 7800
                },
            )
        except Exception as e:
            import json, traceback
            print("❌ Bedrock converse FAILED:", repr(e))
            try:
                err = getattr(e, "response", {}).get("Error", {})
                if err:
                    print("↪︎ Code:", err.get("Code"))
                    print("↪︎ Message:", err.get("Message"))
                md = getattr(e, "response", {}).get("ResponseMetadata", {})
                if md:
                    print("↪︎ RequestId:", md.get("RequestId"))
                    print("↪︎ HTTPStatusCode:", md.get("HTTPStatusCode"))
            except Exception:
                pass
            try:
                print("↪︎ Sent messages:", json.dumps(messages)[:4000])
                print("↪︎ toolConfig:", json.dumps(tool_config))
                print("↪︎ toolChoice:", json.dumps({"tool": {"name": TOOL_NAME}}))
            except Exception:
                pass
            traceback.print_exc()
            resp = None  # graceful fallback

        summ, df_val, graph = "", [], ""
        if resp:
            blocks = (resp.get("output", {}) or {}).get("message", {}).get("content", []) or []
            tool_blocks = [b for b in blocks if "toolUse" in b]
            if tool_blocks:
                tu = tool_blocks[0]["toolUse"]
                payload = tu.get("input", {}) or {}
                summ  = str(payload.get("summ", "")).strip()
                df_in = payload.get("df", [])
                df_val = df_in if isinstance(df_in, list) else []
                graph = str(payload.get("graph", "")).strip()
            else:
                txts = [b.get("text") for b in blocks if "text" in b and b.get("text")]
                if txts:
                    raw = txts[0].strip().strip("```json").strip("```").strip()
                    try:
                        payload = json.loads(raw)
                        summ  = str(payload.get("summ",""))
                        df_in = payload.get("df", [])
                        df_val = df_in if isinstance(df_in, list) else []
                        graph = str(payload.get("graph",""))
                    except Exception:
                        pass

        node_data = {
            "_id": ObjectId(request.node_id),
            "flow_id": ObjectId(request.flow_id),
            "parent_node_ids": request.parent_node_ids,
            "question": request.question,
            "summ": summ,
            "df": validate_dataframe(df_val),
            "graph": graph,
            "type": "MultipleQA",
            "is_delete": "false",
            "created_at": datetime.datetime.utcnow(),
        }
        node_collection.insert_one(node_data)

        return MultipleQuestionAnswerQueryResponse(
            data={
                "question": request.question,
                "summ": summ,
                "df": validate_dataframe(df_val),
                "graph": graph,
                "flow_id": request.flow_id,
                "parent_node_ids": request.parent_node_ids,
                "component_type": "MultipleQA",
            },
            id=request.node_id,
            type="MultipleQA",
            parent_node_ids=request.parent_node_ids,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in /multiple-qa-summarize: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing the request: {str(e)}"
        )

@app.post("/flow-summarizer", response_model=FlowSummarizeResponse)
def flow_summarizer(request: FlowSummarizeRequest):
    try:
        cursor = node_collection.find({"flow_id": ObjectId(request.flow_id)})
        nodes = list(cursor)
        if not nodes:
            raise HTTPException(status_code=404, detail="No nodes found for the given flow_id.")

        conversation = []
        print("These are nodes for flow:", request.flow_id)
        for node in nodes:
            node_id = node["_id"]
            print("node_id:", node_id)
            q, a = fetch_question_answer_from_node_collection(node_id, request.flow_id)
            if q:
                conversation.append({"role": "user", "content": q})
            if a:
                conversation.append({"role": "assistant", "content": f"Answer:\n{a}"})

        print("Conversation turns:", conversation)

        TOOL_NAME = "JSXReport"
        tool_config = {
            "tools": [{
                "toolSpec": {
                    "name": TOOL_NAME,
                    "description": "Return ONLY JSX string for a report component",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "jsx": {"type": "string"}
                            },
                            "required": ["jsx"],
                            "additionalProperties": False
                        }
                    }
                }
            }],
            "toolChoice": {"tool": {"name": TOOL_NAME}}
        }

        sys_text = (
            "You are an AI assistant tasked with generating a JSX element for a professional report.\n"
            "Return a SINGLE JSXReport tool call ONLY (no plain text). The tool input must contain a single key 'jsx' with the JSX string.\n\n"
            "Rules:\n"
            "1) Return ONLY valid JSX. No explanations/comments/backticks.\n"
            "2) Use only plotly.js and ag-grid-community. No other libs.\n"
            "3) The output should look like a structured report, not just 2-3 components.\n"
            "4) Sections:\n"
            "   - Executive Summary (100–150 words)\n"
            "   - Key Metrics\n"
            "   - Performance Tables (use multiple Ag-Grid tables)\n"
            "   - Multiple Charts for Trends (Plotly)\n"
            "   - Additional Insights\n"
            "   - Summary should be long as one page\n"
            "5) Professional styling and structured formatting.\n"
            "6) Use Ag-Grid for multiple tables with ref, rowClass, rowHeight, rowStyle, headerHeight, domLayout=autoHeight.\n"
            "7) Use Plotly for multiple charts; DO NOT style plots beyond layout if needed.\n"
            "8) Background of main div: white. p/h1 font color: black.\n"
            "9) Make layout responsive. Use gaps/HRs between sections. Center the report title at the top.\n"
            "10) Include HR after each section; add spacing between Ag-Grid tables.\n"
            "11) IMPORTANT: RETURN ONLY THE JSX via the tool call; no extra text."
        )

        bedrock_history = []
        for turn in conversation:
            role = "assistant" if turn.get("role") == "assistant" else "user"
            text = (turn.get("content") or "").strip()
            if text:
                bedrock_history.append({"role": role, "content": [{"text": text}]})

        messages = [
            {"role": "user", "content": [{"text": sys_text}]},
            *bedrock_history
        ]

        try:
            resp = bedrock.converse(
                modelId="amazon.nova-lite-v1:0",
                messages=messages,
                toolConfig=tool_config,
                inferenceConfig={
                    "temperature": 0.2,
                    "topP": 0.9,
                    "maxTokens": 9800
                },
            )
            print("Bedrock response:", resp)
        except Exception as e:
            import json, traceback
            print("❌ Bedrock converse FAILED:", repr(e))
            try:
                err = getattr(e, "response", {}).get("Error", {})
                if err:
                    print("↪︎ Code:", err.get("Code"))
                    print("↪︎ Message:", err.get("Message"))
                md = getattr(e, "response", {}).get("ResponseMetadata", {})
                if md:
                    print("↪︎ RequestId:", md.get("RequestId"))
                    print("↪︎ HTTPStatusCode:", md.get("HTTPStatusCode"))
            except Exception:
                pass
            try:
                print("↪︎ Sent messages (truncated):", json.dumps(messages)[:3000])
                print("↪︎ toolConfig:", json.dumps(tool_config))
            except Exception:
                pass
            traceback.print_exc()
            resp = None

        jsx = ""
        if resp:
            blocks = (resp.get("output", {}) or {}).get("message", {}).get("content", []) or []
            tool_blocks = [b for b in blocks if "toolUse" in b]
            if tool_blocks:
                tu = tool_blocks[0]["toolUse"]
                payload = tu.get("input", {}) or {}
                jsx = str(payload.get("jsx", "")).strip()
            else:
                # Fallback: parse text as JSX (strip code fences if any)
                txts = [b.get("text") for b in blocks if "text" in b and b.get("text")]
                if txts:
                    jsx = txts[0].strip().removeprefix("```jsx").removeprefix("```").removesuffix("```").strip()

        flat_jsx = jsx.replace("\n", "")

        flow_collection.update_one(
            {"_id": ObjectId(request.flow_id)},
            {"$set": {"summary": flat_jsx}}
        )

        resp_payload = FlowSummarizeResponse(
            flow_id=request.flow_id,
            response=flat_jsx,
        )
        print("flow_summarizer response:", resp_payload)
        return resp_payload

    except Exception as e:
        print(f"Error in /flow-summarizer: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing the request: {str(e)}"
        )
