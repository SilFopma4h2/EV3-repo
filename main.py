#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import time
import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib

# =========================
# Config
# =========================
MAX_LEN = 32
ROUNDS = 64
LEARNING_RATE = 0.001
JSON_FILE = "ml_hash_log.json"
MODEL_FILE = "ml_hash_model.pkl"

# ANSI kleuren voor terminal
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# =========================
# Helpers
# =========================
def preprocess(text):
    nums = [ord(c) for c in text]
    if len(nums) > MAX_LEN:
        nums = nums[:MAX_LEN]
    else:
        nums += [0]*(MAX_LEN-len(nums))
    return np.array(nums, dtype=float).reshape(1, -1)  # 2D voor sklearn

def bit_diff(a, b):
    return bin(a ^ b).count("1")

def compress(state):
    x = int(np.sum(state[:len(state)//2])) & 0xFFFFFFFF
    y = int(np.sum(state[len(state)//2:])) & 0xFFFFFFFF
    return x, y

# =========================
# Load or init ML-model
# =========================
if os.path.exists(MODEL_FILE):
    ml_model = joblib.load(MODEL_FILE)
    print(f"{Colors.OKGREEN}ML-model geladen van schijf{Colors.ENDC}")
else:
    ml_model = MLPRegressor(hidden_layer_sizes=(16,), activation='relu',
                            solver='sgd', learning_rate_init=LEARNING_RATE,
                            max_iter=1, warm_start=True)
    ml_model.fit(np.zeros((1, MAX_LEN)), np.zeros((1, MAX_LEN)))
    print(f"{Colors.WARNING}Nieuw ML-model aangemaakt{Colors.ENDC}")

# =========================
# Hash functie
# =========================
def ml_hash(text):
    block = preprocess(text)
    state = block.copy()
    
    for _ in range(ROUNDS):
        new_state = ml_model.predict(state)
        state = (state + new_state) % 256
    
    x, y = compress(state.flatten())
    return x, y, f"{x:08x}{y:08x}", state.flatten()

# =========================
# Online learning
# =========================
def learn(prev_state, new_state):
    if prev_state is None:
        return 0
    diff = np.sum(np.abs(new_state - prev_state))
    target = np.random.rand(*new_state.shape)*255  # educatief doel
    ml_model.partial_fit(prev_state.reshape(1, -1), target.reshape(1, -1))
    return diff

# =========================
# Main loop
# =========================
def main():
    log = []
    prev_state = None

    print(f"{Colors.HEADER}ML Hash Terminal Interface (educatief){Colors.ENDC}")
    print("Typ tekst om te hashen. Leeg = stoppen.\n")

    while True:
        text = input(f"{Colors.OKBLUE}> {Colors.ENDC}")
        if not text:
            break

        start = time.time()
        x, y, h, state = ml_hash(text)
        score = learn(prev_state, state)
        duration = time.time() - start

        entry = {
            "input": text,
            "hash": h,
            "avalanche_score": float(score),
            "time_ms": round(duration*1000,3)
        }
        log.append(entry)
        prev_state = state.copy()

        print(f"{Colors.OKGREEN}Hash: {h}{Colors.ENDC}")
        print(f"Avalanche score: {score:.2f}")
        print(f"Tijd: {duration*1000:.2f} ms")
        print("-"*50)

    # Opslaan JSON en ML-model
    with open(JSON_FILE, "w") as f:
        json.dump(log, f, indent=4)
    joblib.dump(ml_model, MODEL_FILE)

    print(f"\n{Colors.BOLD}Alles opgeslagen!{Colors.ENDC}")
    print(f"JSON: {JSON_FILE}")
    print(f"ML-model: {MODEL_FILE}")

if name == "__main__":
    main()
