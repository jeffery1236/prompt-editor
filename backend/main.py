import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from prompt_initialization import PromptInitializer
from prompt_segmentation import get_segmented_prompt, SECTION_TYPES
from general_feedback import get_general_feedback, GeneralFeedback
from in_text_feedback import InTextFeedback, get_in_text_feedback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; you can change this to specific domains
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

class PromptImprovementFeedback(BaseModel):
    inTextFeedback: list[InTextFeedback]
    generalFeedback: list[GeneralFeedback]

class PromptImprovementInputs(BaseModel):
    prompt: str

class PromptInitializationInputs(BaseModel):
    short_task_description: str
    additional_features: dict

class PromptInitializationResponse(BaseModel):
    prompt: str

@app.post(
    "/prompt_improvement",
    operation_id="getPromptImprovement",
    response_model=PromptImprovementFeedback,
)
async def get_prompt_improvement(request_transport: PromptImprovementInputs) -> PromptImprovementFeedback:
    prompt = request_transport.prompt

    prompt_segments = await get_segmented_prompt(prompt)


    # Get General Feedback

    present_sections = set([segment.type.value for segment in prompt_segments if segment.content])
    all_section_types = set([section.value for section in SECTION_TYPES])

    missing_sections = all_section_types - present_sections
    print("Missing sections:", missing_sections)

    # Start general feedback and in-text feedback concurrently
    tasks = []

    if missing_sections:
        tasks.append(get_general_feedback(prompt, missing_sections))
    else:
        tasks.append(asyncio.create_task(asyncio.sleep(0)))  # Add an empty task for the general feedback
    
    tasks.append(get_in_text_feedback(prompt, prompt_segments))
    general_feedback, in_text_feedback = await asyncio.gather(*tasks)

    return PromptImprovementFeedback(
        inTextFeedback=in_text_feedback,
        generalFeedback=general_feedback if missing_sections else []
    )

@app.post(
    "/prompt_initialization",
    operation_id="getPromptInitialization",
    response_model=PromptInitializationResponse,
)
async def get_prompt_initialization(request_transport: PromptInitializationInputs) -> PromptInitializationResponse:
    prompt = await PromptInitializer.get_prompt_initialization_outputs(
        short_task_description=request_transport.short_task_description,
        input_feature_descriptions=request_transport.additional_features
    )

    return PromptInitializationResponse(prompt=prompt)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
