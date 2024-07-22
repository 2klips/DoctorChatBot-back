from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    text: str

# Initialize the model and tokenizer
model_path = 'model_1'  # 모델 경로를 여기에 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
tokenizer = T5TokenizerFast.from_pretrained(model_path)

@app.post('/generate_response')
async def generate_response(question: Question):
    input_text = question.text

    # Tokenize input
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)

    # Generate response
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=1024,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            top_p=1,
        )

    # Decode response
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return {"response": decoded_output}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)