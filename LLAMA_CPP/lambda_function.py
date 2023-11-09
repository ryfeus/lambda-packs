from llama_cpp import Llama

MODEL_SESSION = None

def handler(event, context):
    global MODEL_SESSION
    if MODEL_SESSION is None:
        MODEL_SESSION = Llama(model_path="model/mistral-7b-v0.1.Q5_K_S.gguf")

    instruction = event.get("instruction", "What is the capital of Spain?")
    response = MODEL_SESSION(instruction, max_tokens=32, stop=["Q:", "\n"], echo=True)
    return {"result": response}
