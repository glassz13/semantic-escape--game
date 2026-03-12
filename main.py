import uuid
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict
import game_engine as engine

app = FastAPI()
sessions: Dict[str, engine.GameState] = {}

# request models only
class MoveRequest(BaseModel):
    session_id: str
    word: str

class RiddleOptionRequest(BaseModel):
    session_id: str
    choice: str

class RiddleAnswerRequest(BaseModel):
    session_id: str
    answer: str

# helper
def state_dict(state: engine.GameState) -> dict:
    return {
        "room_name": state.room_name,
        "start_word": state.start_word,
        "target_word": state.target_word,
        "path": state.path,
        "lives": state.lives,
        "min_total": state.min_total,
        "status": state.status,
        "threshold": state.threshold,
        "current_room_index": state.current_room_index,
    }

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join("static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found in static/")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/start")
async def start_game():
    session_id = str(uuid.uuid4())
    state = engine.create_game()
    sessions[session_id] = state
    return {"session_id": session_id, **state_dict(state)}

@app.post("/move")
async def move(req: MoveRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    state = sessions[req.session_id]
    if state.status != "playing":
        raise HTTPException(status_code=400, detail=f"Cannot submit word in status: {state.status}")
    success, message, new_state = engine.submit_word(state, req.word)
    sessions[req.session_id] = new_state
    resp = {"success": success, "message": message, **state_dict(new_state)}
    if new_state.status == "awaiting_mandatory_riddle":
        resp["riddle_question"] = new_state.mandatory_riddle_question
        resp["attempts_left"] = new_state.mandatory_riddle_attempts
    return resp

@app.post("/riddle-option")
async def riddle_option(req: RiddleOptionRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    state = sessions[req.session_id]
    if state.status != "room_complete":
        raise HTTPException(status_code=400, detail="No riddle available now")
    if req.choice.lower() in ("y", "yes"):
        new_state = engine.offer_optional_riddle(state)
        sessions[req.session_id] = new_state
        return {"riddle_question": new_state.optional_riddle_question, **state_dict(new_state)}
    else:
        new_state = engine.skip_optional_riddle(state)
        sessions[req.session_id] = new_state
        return state_dict(new_state)

@app.post("/riddle-answer")
async def riddle_answer(req: RiddleAnswerRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    state = sessions[req.session_id]
    if state.status == "awaiting_riddle":
        correct, msg, new_state = engine.answer_optional_riddle(state, req.answer)
        sessions[req.session_id] = new_state
        return {"correct": correct, "message": msg, **state_dict(new_state)}
    elif state.status == "awaiting_mandatory_riddle":
        correct, attempts_left, msg, new_state = engine.answer_mandatory_riddle(state, req.answer)
        sessions[req.session_id] = new_state
        return {"correct": correct, "message": msg, "attempts_left": attempts_left, **state_dict(new_state)}
    else:
        raise HTTPException(status_code=400, detail="No pending riddle")

@app.get("/state/{session_id}")
async def get_state(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return engine.get_room_info(sessions[session_id])

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)