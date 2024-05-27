import asyncio
import codecs
from http import HTTPStatus
import tempfile
import json
import time
import os
from pathlib import Path
from typing import (
    AnyStr, List, Optional, final, TypedDict, 
    Iterable, Awaitable, Tuple, Union
)

from fastapi import Request, UploadFile

from openai.types.chat import ChatCompletionRole, ChatCompletionContentPartParam

from vllm.config import ModelConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import ( 
    ChatCompletionRequest, ChatCompletionResponse,
    ChatMessage, DeltaMessage, ErrorResponse,
    BatchRequestOutputObject, CreateBatchRequest, 
    BatchObject 
)
from vllm.model_executor.guided_decoding import (
    get_guided_decoding_logits_processor
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath, OpenAIServing
from vllm.logger import init_logger
from vllm.utils import random_uuid

logger = init_logger(__name__)

@final
class ConversationMessage(TypedDict):
    role: str
    content: str


class File:
    """
    Empty named temporary file. All files created are not deleted out of scope.
    Manual deletion is required when no longer in use.
    """
    def __init__(self, purpose, filename = None) :
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.id: str = f"file-{random_uuid()}"
        self.purpose: str = purpose
        self.size: int = 0 # Size in bytes
        self.creation: int = int(time.time())
        if not filename:
            filename = f"tempfile-{random_uuid()}"
        self.filename: str = filename

    def write(self, bytes: AnyStr, json_format: bool = False) -> int:
        """
        Validates json format and writes it to file

        returns: Number of bytes written
        """
        
        # Validate
        if isinstance(bytes, str):
            bytes = bytes.encode("utf-8")

        try:
            if json_format:
                lines = bytes.decode("utf-8").split("\n")
                for line in lines:
                    if line: # Skip empty lines
                        json.loads(line)
        except Exception as e:
            logger.critical(f"{e}. Provided file format is incorrect. Currently only jsonl is supported.")
            return 0
        else:
            self.size += len(bytes)
            bytes_written = self.file.write(bytes)
            return bytes_written

    def read(self) -> bytes:
        """
        Reads bytes from the file
        """
        self.file.seek(0)
        return self.file.read()
    
    def fetch_lines(self):
        """
        Generator to return each jsonl
        """
        self.file.seek(0)
        while line:=self.file.readline():
            if line != b'\n':
                yield line 
    
    def close(self):
        """
        Close the temporary file.
        """
        self.file.close()

    def jsonify(self):
        return dict(
            id=self.id,
            object='file',
            bytes=self.size,
            created_at=self.creation,
            filename=self.filename,
            purpose=self.purpose
        )


class BatchHandler:
    handler = None
    tmp_files: dict = dict()
    batch_data: dict = dict()
    batch_tasks: set = set()

    def __init__(self, max_files: int = 100, max_size: int = 104857600) -> None:
        """
        max_files: Maximum number of files that can be created in temporary directory
        max_size: Maximum size of each file in bytes
        """
        if max_files:
            self.max_files = max_files
        if max_size:
            self.max_size = max_size

    @classmethod
    def get_handler(cls):
        if cls.handler:
            return cls.handler
        else:
            cls.handler = BatchHandler()
            return cls.handler
    
    @staticmethod
    def set_tmp_dir(dir_path):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        tempfile.tempdir = dir_path

    def has_file(self, file_id):
        return file_id in self.tmp_files

    def create_tmp_file(self, purpose: str, file: UploadFile = None):
        if len(self.tmp_files) > self.max_files:
            raise Exception("Maximum number of files created. Clean up existing files")

        save_file = File(purpose)
        if file:
            if file.size > self.max_size:
                raise Exception("File size too large. Consider making a smaller file")
            save_file.filename = file.filename 
            bytes_written = save_file.write(file.file.read(), json_format=True)
            if bytes_written != file.size:
                return None

        self.tmp_files[save_file.id] = save_file
        return save_file
    
    def handle_errored_request(self, batch_id):
        batch = self.batch_data[batch_id]
        batch[3][1] += 1
        batch[0].request_counts["failed"] += 1

    def handle_success_request(self, batch_id):
        batch = self.batch_data[batch_id]
        batch[2][1] += 1
        batch[0].request_counts["completed"] += 1

    def check_batch_complete(self, batch_id: str):
        return self.batch_data[batch_id].status == "complete"
    
    def get_batch(self, batch_id):
        return self.batch_data[batch_id]

    def get_all_batches(self):
        return self.batch_data

    def cancel_batch(self, batch_id):
        if batch_id not in self.batch_data:
            return False
        batch = self.batch_data[batch_id]
        output_file = batch.output_file_id
        error_file = batch.error_file_id

        output_file = self.tmp_files.pop(output_file)
        error_file = self.tmp_files.pop(error_file)

        output_file.close()
        error_file.close()

        batch.cancelled_at = int(time.time())
        batch.status = "cancelled"

        task = [task for task in self.batch_tasks if task.name == batch_id][0]
        self.batch_tasks.discard(task)

    
    def _batch_complete_callback(self, batch_task):
        """
        Set the status of completed batch. Also perform cleanup
        """
        self.batch_tasks.discard(batch_task)

    def _count_requests(self, file: File):
        data = file.read().strip()
        data = data.decode("utf-8").split('\n')
        return len(data)

    def create_batch(self, batch: BatchObject):

        input_file = self.tmp_files[batch.input_file_id]
        output_file = self.tmp_files[batch.output_file_id]
        error_file = self.tmp_files[batch.error_file_id]

        # batch data stores the batch object and associated files
        # along with the number of requests/response in each line

        self.batch_data[batch.id] = (batch, 
                                     [input_file, self._count_requests(input_file)],
                                     [output_file, 0],
                                     [error_file, 0]
        )

        batch.request_counts["total"] = self._count_requests(input_file)

    def add_batch_task(self, task):
        self.batch_tasks.add(task)


class OpenAIServingFiles(OpenAIServing):

    def __init__(self):
        self.handler = BatchHandler.get_handler()
        cur_dir = os.getcwd()
        tmp_dir = cur_dir + '/tmp'
        BatchHandler.set_tmp_dir(tmp_dir)
    
    async def create_file(self, file: UploadFile, purpose: str):
        try:
            save_file = self.handler.create_tmp_file(purpose, file=file)
        except Exception as e:
            return self.create_error_response(
                message=f"{e}",
                err_type="BadRequestError",
                status_code=HTTPStatus.BAD_REQUEST
            )
        if not save_file:
            return None
        return save_file.jsonify()

    async def list_files(self, purpose: str = None):
        data = [file.jsonify() for file in self.handler.tmp_files.values()]
        
        # reduce if specific purpose  
        if purpose:  
            data = [file for file in data if file['purpose'] == purpose]

        return dict(
            data=data,
            object='list'
        )
    
    async def retrieve_file(self, file_id: str):
        file = self.handler.tmp_files.get(file_id)
        if not file:
            return None
    
        return file.jsonify()

    async def delete_file(self, file_id: str):
        deleted = False
        if file_id in self.handler.tmp_files:
            del self.handler.tmp_files[file_id] 
            deleted = True
        
        return dict(
            id=file_id,
            object='file',
            deleted=deleted
        )
    
    async def retrieve_file_content(self, file_id: str):
        if file_id in self.handler.tmp_files:
            file = self.handler.tmp_files.get(file_id)
            return file.read()
        else:
            return None

class OpenAIServingBatch(OpenAIServingChat):
    
    def __init__(self,
                 engine: AsyncLLMEngine,
                 model_config: ModelConfig,
                 served_model_names: List[str],
                 response_role: str,
                 lora_modules: Optional[List[LoRAModulePath]] = None,
                 chat_template: Optional[str] = None):
        super().__init__(engine=engine,
                         model_config=model_config,
                         served_model_names=served_model_names,
                         response_role=response_role,
                         lora_modules=lora_modules)

        self.response_role = response_role
        self._load_chat_template(chat_template)

    def create_error_batch_output(self, request_output, request_id, status_code, model, error_msg):
        response = dict(
                        status_code=status_code,
                        request_id=request_id,
                        body=ChatCompletionResponse(model=model,
                                                    choices=[],
                                                    usage=dict())
                    )
        request_output.response = response
        request_output.error = dict(
            code=1, # Internal Error Code
            message=f"{error_msg}"
        )

        return request_output

    async def get_batches(self):
        handler = BatchHandler.get_handler()
        batches = handler.get_all_batches()

        return dict(
            object="list",
            data=[ batch.model_dump() for batch in batches.values() ]
        )

    async def retrieve_batch(self, batch_id):
        handler = BatchHandler.get_handler()
        return handler.get_batch(batch_id)[0].model_dump()

    async def exec_batch(self, batch_id):
        handler = BatchHandler.get_handler()
        try:
            batch, input_info, output_info, error_info \
                = handler.get_batch(batch_id)

            batch.status = "processing" 

            input_file = input_info[0]
            output_file = output_info[0]
            error_file = error_info[0]

            for request in input_file.fetch_lines():
                request_output = BatchRequestOutputObject()
                try:
                    request = json.loads(request)
                    custom_id = request.get("custom_id")
                    method = request.get("method")
                    url = request.get("url")
                    request_id = f"req_{random_uuid()}"
                    body = request.get("body")
                    request = ChatCompletionRequest(**body)

                    request_output.custom_id = custom_id

                    if not custom_id or not method or not url:
                        raise Exception("Incorrectly formatted request")

                    try:
                        conversation: List[ConversationMessage] = []

                        for m in request.messages:
                            messages, _ = self._parse_chat_message_content(
                            m["role"], m["content"])

                            conversation.extend(messages)

                        prompt = self.tokenizer.apply_chat_template(
                            conversation=conversation,
                            tokenize=False,
                            add_generation_prompt=request.add_generation_prompt,
                        )
                    except Exception as e:
                        logger.error("Error in applying chat template from request: %s", e)
                        request_output = self.create_error_batch_output(request_output, 
                                                       request_id,
                                                       HTTPStatus.BAD_REQUEST,
                                                       body.get("model"),
                                                       e)
                        error_file.write(request_output.model_dump_json() + "\n")
                        handler.handle_errored_request(batch_id)

                    cmpl_req_id = f"chatcmpl-{random_uuid()}"
                    try:
                        # Tokenize/detokenize depending on prompt format (string/token list)
                        prompt_ids, prompt_text = self._validate_prompt_and_tokenize(
                            request, prompt=prompt)
                        sampling_params = request.to_sampling_params()
                        lora_request = self._maybe_get_lora(request)
                        decoding_config = await self.engine.get_decoding_config()
                        guided_decoding_backend = request.guided_decoding_backend \
                            or decoding_config.guided_decoding_backend
                        guided_decode_logits_processor = (
                            await get_guided_decoding_logits_processor(
                                guided_decoding_backend, request, await
                                self.engine.get_tokenizer()))
                        if guided_decode_logits_processor:
                            if sampling_params.logits_processors is None:
                                sampling_params.logits_processors = []
                            sampling_params.logits_processors.append(
                                guided_decode_logits_processor)
                    except ValueError as e:
                        request_output = self.create_error_batch_output(request_output, 
                                                       request_id,
                                                       HTTPStatus.INTERNAL_SERVER_ERROR,
                                                       body.get("model"),
                                                       e)
                        error_file.write(request_output.model_dump_json() + "\n")
                        handler.handle_errored_request(batch_id)

                    result_generator = self.engine.generate(prompt_text, sampling_params,
                                                            cmpl_req_id, prompt_ids,
                                                            lora_request)
                    try:
                        raw_request = None
                        cmpl_response = await self.chat_completion_full_generator(
                            request, raw_request, result_generator, cmpl_req_id,
                            conversation)
                        request_output.response = dict(
                            status_code=200,
                            request_id=cmpl_req_id,
                            body=cmpl_response
                        )
                        output_file.write(request_output.model_dump_json() + "\n")
                        handler.handle_success_request(batch_id)

                    except ValueError as e:
                        request_output = self.create_error_batch_output(request_output, 
                                                       request_id,
                                                       HTTPStatus.INTERNAL_SERVER_ERROR,
                                                       body.get("model"),
                                                       e)
                        error_file.write(request_output.model_dump_json() + "\n")
                        handler.handle_errored_request(batch_id)

                except Exception as e:
                    request_output = self.create_error_batch_output(request_output, 
                                                        request_id,
                                                        HTTPStatus.BAD_REQUEST,
                                                        body.get("model"),
                                                        e)
                    error_file.write(request_output.model_dump_json() + "\n")
                    handler.handle_errored_request(batch_id)

            batch.status = "completed"
            batch.completed_at = int(time.time())
        
        except asyncio.CancelledError:
            handler.cancel_batch(batch_id)

    async def create_batch_chat_completion(self, request: CreateBatchRequest, raw_request: Request
                                           ) -> Union[ ErrorResponse, BatchObject]:

        """ Batch processing API similar to OpenAI's API """ 

        handler = BatchHandler.get_handler()
        input_file = request.input_file_id

        if not handler.has_file(input_file):
            return self.create_error_response(
                message="Provided input file is not present "\
                    "on the server",
                err_type="NotFoundError",
                status_code=HTTPStatus.NOT_FOUND)
        
        if request.endpoint != "/v1/chat/completions":
            return self.create_error_response(
                message="Only chat completion endpoint is availbale for batch API",
                err_type="BadRequestError",
                status_code=HTTPStatus.BAD_REQUEST
            )

        if request.completion_window != "24h":
            return self.create_error_response(
                message="Only 24h window is available for Batch API",
                err_type="BadRequestError",
                status_code=HTTPStatus.BAD_REQUEST
            )
        
        output_file = handler.create_tmp_file("output_file")
        error_file = handler.create_tmp_file("error_file")

        batch = BatchObject(
            created_at=int(time.time()),
            completed_at=None,
            cancelled_at=None,
            endpoint=request.endpoint,
            input_file_id=request.input_file_id,
            completion_window=request.completion_window,
            status="created",
            output_file_id=output_file.id,
            error_file_id=error_file.id,
            request_counts=dict(
                total=0,
                completed=0,
                failed=0            
            )
        )

        handler.create_batch(batch)

        task = asyncio.create_task(self.exec_batch(batch.id))
        task.set_name(batch.id)
        task.add_done_callback(handler._batch_complete_callback)
        handler.add_batch_task(task)

        return batch.model_dump()