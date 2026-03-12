import random
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
_CACHE = {}

def embed(word: str):
    k = word.lower()
    if k in _CACHE:
        return _CACHE[k]
    _CACHE[k] = _MODEL.encode([k])[0]
    return _CACHE[k]

def sim(a: str, b: str) -> float:
    va, vb = embed(a), embed(b)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))

def valid_word(w: str) -> bool:
    return isinstance(w, str) and w.isalpha()

def clean(s: str) -> str:
    return "".join(c for c in (s or "").lower() if c.isalpha())

ROOMS = [
    {"name": "The Wild Dominion", "pairs": [
        ("dog", "wolf"),
        ("fish", "shark"),
        ("cat", "lion"),
        ("pet", "beast"),
        ("bird", "eagle"),
        ("pond", "ocean"),
    ], "thr": 0.35, "min_total": 5},

    {"name": "The Mortal Sphere", "pairs": [
        ("child", "soldier"),
        ("home", "ruin"),
        ("smile", "scar"),
        ("cradle", "grave"),
        ("dream", "regret"),
        ("laugh", "mourn"),
    ], "thr": 0.36, "min_total": 5},

    {"name": "The Living World", "pairs": [
        ("rain", "drought"),
        ("seed", "wildfire"),
        ("river", "desert"),
        ("bloom", "frost"),
        ("spring", "winter"),
        ("leaf", "ash"),
    ], "thr": 0.37, "min_total": 5},

    {"name": "The Human Paradox", "pairs": [
        ("love", "war"),
        ("hope", "despair"),
        ("hero", "tyrant"),
        ("freedom", "chains"),
        ("mercy", "vengeance"),
        ("brother", "enemy"),
    ], "thr": 0.37, "min_total": 6},

    {"name": "The Eternal Gate", "pairs": [
        ("god", "void"),
        ("light", "abyss"),
        ("soul", "dust"),
        ("first", "last"),
        ("eternity", "silence"),
        ("dawn", "oblivion"),
    ], "thr": 0.38, "min_total": 6},
]
RIDDLES = {
    "The Wild Dominion": [
        ("I have a mane but I'm not a horse, I roar but I'm not thunder. What am I?", ["lion", "beast"]),
        ("I build without hands, I tunnel without tools, I live underground. What am I?", ["mole", "worm"]),
        ("I have eight legs and spin silk, but I am no seamstress. What am I?", ["spider"]),
        ("I have no voice yet I warn the herd, I have no legs yet I move the trees. What am I?", ["wind", "breeze"]),
        ("I am hunted for my coat but I wear it better than any king. What am I?", ["fox", "leopard"]),
        ("I drink from rivers but I am not thirsty, I kill but I am not angry. What am I?", ["crocodile", "predator"]),
    ],
    "The Mortal Sphere": [
        ("I have hands but cannot clap, I tell time but cannot speak. What am I?", ["clock", "watch"]),
        ("I am passed from father to son but owned by neither. What am I?", ["name", "legacy"]),
        ("The more you take, the more you leave behind. What am I?", ["footsteps", "steps"]),
        ("I have no weight but can break a man. What am I?", ["silence", "grief"]),
        ("I start as joy and end as memory. What am I?", ["life", "childhood"]),
        ("I am built by two and destroyed by one. What am I?", ["trust", "bond"]),
    ],
    "The Living World": [
        ("I drink water but I am not thirsty, I eat light but I have no mouth. What am I?", ["plant", "tree"]),
        ("I fall from the sky but I am not rain, I blanket the earth but give no warmth. What am I?", ["snow", "ash"]),
        ("I begin as a stone prison, then I crack open and life walks out. What am I?", ["egg", "seed"]),
        ("I am born in water, I live on land, I return to water to die. What am I?", ["salmon", "frog"]),
        ("I have no lungs but I breathe, I have no mouth but I feed millions. What am I?", ["forest", "earth"]),
        ("I die every night and am reborn every morning. What am I?", ["sun", "flower"]),
    ],
    "The Human Paradox": [
        ("The man who makes it doesn't need it, the man who buys it doesn't use it, the man who uses it doesn't know it. What is it?", ["coffin", "grave"]),
        ("I can be cracked, made, told and played. What am I?", ["joke", "game"]),
        ("I have cities but no houses, forests but no trees, rivers but no water. What am I?", ["map", "painting"]),
        ("The more you share me the more you have of me. What am I?", ["knowledge", "love"]),
        ("I am the thing men die for but cannot hold. What am I?", ["glory", "freedom"]),
        ("I grow stronger when broken. What am I?", ["trust", "character"]),
    ],
    "The Eternal Gate": [
        ("I was before time, I will outlast the stars, yet I fit inside a single moment. What am I?", ["eternity", "infinity"]),
        ("I never was, am always to be. No one has ever seen me. What am I?", ["tomorrow", "future"]),
        ("What disappears the instant you speak its name?", ["silence", "quiet"]),
        ("I have no beginning, no middle, no end. I am not emptiness yet I hold everything. What am I?", ["void", "universe"]),
        ("I am the last sound and the first silence. What am I?", ["death", "end"]),
        ("Every god has feared me, every man has met me. What am I?", ["death", "void"]),
    ],
}

class GameState:
    def __init__(self, current_room_index=0, lives=3, path=None, current_word=None,
                 start_word=None, target_word=None, threshold=0.0, min_total=5,
                 room_name="", status="playing", optional_riddle_question=None,
                 optional_riddle_answers=None, mandatory_riddle_attempts=3,
                 mandatory_riddle_question=None, mandatory_riddle_answers=None):
        self.current_room_index = current_room_index
        self.lives = lives
        self.path = path if path is not None else []
        self.current_word = current_word
        self.start_word = start_word
        self.target_word = target_word
        self.threshold = threshold
        self.min_total = min_total
        self.room_name = room_name
        self.status = status
        self.optional_riddle_question = optional_riddle_question
        self.optional_riddle_answers = optional_riddle_answers
        self.mandatory_riddle_attempts = mandatory_riddle_attempts
        self.mandatory_riddle_question = mandatory_riddle_question
        self.mandatory_riddle_answers = mandatory_riddle_answers

    def copy(self, **kwargs):
        attrs = vars(self).copy()
        attrs.update(kwargs)
        return GameState(**attrs)

def create_game() -> GameState:
    room = ROOMS[0]
    start, target = random.choice(room["pairs"])
    return GameState(
        current_room_index=0, lives=3, path=[start], current_word=start,
        start_word=start, target_word=target, threshold=room["thr"],
        min_total=room["min_total"], room_name=room["name"], status="playing",
    )

def submit_word(state: GameState, word: str):
    word_clean = clean(word)
    if not valid_word(word_clean):
        return False, "Invalid input. Use a single alphabetic word.", state
    if word_clean in state.path:
        return False, "Word already used in path.", state
    if word_clean == state.target_word and (len(state.path) + 1) < state.min_total:
        remaining = state.min_total - len(state.path) - 1
        return False, f"Need {remaining} more bridge word{'s' if remaining != 1 else ''} before reaching the target.", state

    s_prev = sim(state.current_word, word_clean)
    if s_prev < state.threshold:
        new_lives = state.lives - 1
        if new_lives <= 0:
            return False, f"Rejected — link strength: {s_prev:.2f}", state.copy(lives=0, status="game_over")
        return False, f"Rejected — link strength: {s_prev:.2f}", state.copy(lives=new_lives)

    new_path = state.path + [word_clean]
    new_state = state.copy(path=new_path, current_word=word_clean)

    if word_clean == state.target_word:
        if state.current_room_index == len(ROOMS) - 1:
            q, ans = random.choice(RIDDLES[state.room_name])
            return True, "Gate unlocked! Answer the final riddle to escape.", new_state.copy(
                status="awaiting_mandatory_riddle",
                mandatory_riddle_question=q,
                mandatory_riddle_answers=ans,
                mandatory_riddle_attempts=3,
            )
        return True, "Gate unlocked!", new_state.copy(status="room_complete")

    return True, f"Accepted — link strength: {s_prev:.2f}", new_state

def offer_optional_riddle(state: GameState) -> GameState:
    q, ans = random.choice(RIDDLES[state.room_name])
    return state.copy(status="awaiting_riddle", optional_riddle_question=q, optional_riddle_answers=ans)

def skip_optional_riddle(state: GameState) -> GameState:
    return _advance_to_next_room(state, next_lives=3)

def answer_optional_riddle(state: GameState, answer: str):
    ans_clean = clean(answer)
    correct = any(ans_clean == clean(v) for v in state.optional_riddle_answers)
    if correct:
        return correct, "Correct! Next realm starts with 5 lives.", _advance_to_next_room(state, 5)
    return correct, "Wrong. Next realm starts with 2 lives.", _advance_to_next_room(state, 2)

def answer_mandatory_riddle(state: GameState, answer: str):
    ans_clean = clean(answer)
    correct = any(ans_clean == clean(v) for v in state.mandatory_riddle_answers)
    attempts_left = state.mandatory_riddle_attempts - 1
    if correct:
        return True, attempts_left, "The final seal shatters. You have escaped!", state.copy(status="escaped")
    if attempts_left <= 0:
        return False, 0, "No attempts left. Game Over.", state.copy(status="game_over")
    return False, attempts_left, f"Wrong. Attempts left: {attempts_left}", state.copy(mandatory_riddle_attempts=attempts_left)

def _advance_to_next_room(state: GameState, next_lives: int) -> GameState:
    next_idx = state.current_room_index + 1
    if next_idx >= len(ROOMS):
        return state.copy(status="escaped")
    room = ROOMS[next_idx]
    start, target = random.choice(room["pairs"])
    return state.copy(
        current_room_index=next_idx, lives=next_lives, path=[start],
        current_word=start, start_word=start, target_word=target,
        threshold=room["thr"], min_total=room["min_total"],
        room_name=room["name"], status="playing",
        optional_riddle_question=None, optional_riddle_answers=None,
    )

def get_room_info(state: GameState) -> dict:
    info = {
        "room_name": state.room_name, "start_word": state.start_word,
        "target_word": state.target_word, "path": state.path,
        "lives": state.lives, "min_total": state.min_total,
        "status": state.status, "current_word": state.current_word,
        "current_room_index": state.current_room_index, "threshold": state.threshold,
    }
    if state.status == "awaiting_riddle":
        info["riddle_question"] = state.optional_riddle_question
    elif state.status == "awaiting_mandatory_riddle":
        info["riddle_question"] = state.mandatory_riddle_question
        info["attempts_left"] = state.mandatory_riddle_attempts
    return info