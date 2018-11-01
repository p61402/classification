#!/usr/bin/env python
# coding: utf-8

# 載入函式庫 🙄

import pandas as pd
import numpy as np
import random


# 定義 attributes 以及對應 value 的範圍

data_range = {}
data_range["Avg sleep time"] = (0, 24)
data_range["Avg study time"] = (0, 24)
data_range["Avg video game time"] = (0, 24)
data_range["BMI"] = (19, 35)
data_range["In a relationship"] = ["Yes", "No"]
data_range["Family financial status"] = ["high", "average", "low"]
data_range["GPA"] = (0, 4.3)
data_range["Laptop brand"] = ["Leveno", "HP", "Dell", "Acer", "Asus", "Microsoft", "Apple", "Razer", "MSI", "Others"]
data_range["TOEIC grade"] = (0, 990)
data_range["IQ"] = (100, 180)
data_range["Grade"] = ["80+", "60~80", "60-"]


# Attributes

attributes = list(data_range.keys())


# 定義機率

IQ_probs = ([0.969/50] * 50) + ([0.001] * 31)
GPA_probs = ([0.00001] * 18) + ([0.99982 / 26] * 26)
sleep_probs = ([0] * 3) + [0.02, 0.05, 0.12, 0.21, 0.24, 0.18, 0.09, 0.05, 0.02, 0.02] + ([0] * 12)
study_probs = [0.001, 0.02, 0.1, 0.25, 0.2, 0.15, 0.1, 0.08, 0.05] + ([0.049 / 6] * 6) + ([0] * 10)
game_probs = [0.2, 0.12, 0.18, 0.1] + ([0.4 / 10] * 10) + ([0] * 11)


# 函式：根據機率產生資料

def generate_data():
    sleep = np.random.choice(np.arange(data_range["Avg sleep time"][0], data_range["Avg sleep time"][1] + 1), p=sleep_probs)
    study = np.random.choice(np.arange(data_range["Avg study time"][0], data_range["Avg study time"][1] + 1), p=study_probs)
    game = np.random.choice(np.arange(data_range["Avg video game time"][0], data_range["Avg video game time"][1] + 1), p=game_probs)
    BMI = random.randint(data_range["BMI"][0], data_range["BMI"][1])
    relation = random.choice(data_range["In a relationship"])
    finalcial = random.choice(data_range["Family financial status"])
    GPA = np.random.choice(np.arange(data_range["GPA"][0], data_range["GPA"][1] + 0.1, 0.1), p=GPA_probs)
    laptop = random.choice(data_range["Laptop brand"])
    TOEIC = int(np.random.normal(500, 250, 1))
    if TOEIC % 5 != 0:
        TOEIC -= TOEIC % 5
    if TOEIC < 0:
        TOEIC = 0
    elif TOEIC > 990:
        TOEIC = 990
    IQ = np.random.choice(np.arange(data_range["IQ"][0], data_range["IQ"][1] + 1), p=IQ_probs)
    grade = None
    d = {}
    d["Avg sleep time"] = sleep
    d["Avg study time"] = study
    d["Avg video game time"] = game
    d["BMI"] = BMI
    d["In a relationship"] = relation
    d["Family financial status"] = finalcial
    d["GPA"] = GPA
    d["Laptop brand"] = laptop
    d["TOEIC grade"] = TOEIC
    d["IQ"] = IQ
    d["Grade"] = grade
    
    return d


# 函式：根據規則將資料歸類

def rules():
    d = generate_data()
    if d["IQ"] >= 150 or d["GPA"] >= 3.7:
        d["Grade"] = data_range["Grade"][0]
    elif d["Family financial status"] == "high" and d["Laptop brand"] == "Apple":
        d["Grade"] = data_range["Grade"][0]
    elif d["Avg sleep time"] <= 5 and d["Avg study time"] >= 8 and d["Family financial status"] in ["low", "avergae"]:
        d["Grade"] = data_range["Grade"][0]
    elif d["Avg study time"] >= 5 and d["In a relationship"] == "No":
        d["Grade"] = random.choice(data_range["Grade"])
    elif d["TOEIC grade"] >= 800 and d["Avg sleep time"] <= 6:
        d["Grade"] = data_range["Grade"][1]
    elif d["Avg video game time"] >= 5:
        d["Grade"] = data_range["Grade"][2]
    else:
        d["Grade"] = random.choice(data_range["Grade"])
    return d


# 產生資料

def generate(data_points_numbers = 1000):
    df = pd.DataFrame(columns=attributes)

    class_count = {}
    class_count[data_range["Grade"][0]] = int(data_points_numbers * (1 / len(data_range["Grade"])))
    class_count[data_range["Grade"][1]] = int(data_points_numbers * (1 / len(data_range["Grade"])))
    class_count[data_range["Grade"][2]] = int(data_points_numbers - class_count[data_range["Grade"][0]] - class_count[data_range["Grade"][1]])

    for _ in range(data_points_numbers):
        d = rules()
        stay = True
        while (stay):
            if (class_count[d["Grade"]] != 0):
                class_count[d["Grade"]] -= 1
                stay = False
            else:
                d = rules()
        s = pd.Series(d, index=attributes)
        df = df.append(s, ignore_index=True)

    # 確認 attributes 的 datatype

    numeric_attributes = ["Avg sleep time", "Avg study time", "Avg video game time", "TOEIC grade", "IQ"]
    df[numeric_attributes] = df[numeric_attributes].apply(pd.to_numeric)
    floating_attributes = ["BMI", "GPA"]
    df[floating_attributes] = df[floating_attributes].astype('float64')
    category_attributes = ["In a relationship", "Family financial status", "Laptop brand", "Grade"]
    df[category_attributes] = df[category_attributes].astype('category')

    return df
