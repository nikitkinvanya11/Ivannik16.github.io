# Пример базы: продукт -> калории на 100 г
CALORIE_DB = {
    "apple": 52,
    "banana": 89,
    "orange": 47,
    "pizza": 266,
    "hamburger": 295,
    "hot dog": 290,
    "sandwich": 250,
    "carrot": 41,
    "broccoli": 34,
    "cake": 350,
    "donut": 452,
    "bread": 265,
    "egg": 155,
    "chicken": 239,
    "rice": 130,
    "pasta": 158,
    "potato": 77,
    "tomato": 18,
    "cucumber": 15,
    "cheese": 402,
    "milk": 42,
    "yogurt": 59,
}

# Сопоставление классов YOLO с нашими ключами
# YOLOv8 использует COCO-датасет, где есть такие классы:
YOLO_TO_FOOD = {
    47: "apple",
    48: "banana",
    49: "orange",
    53: "pizza",
    54: "donut",
    55: "cake",
    52: "hot dog",
    50: "broccoli",
    51: "carrot",
}

# Обратное сопоставление: имя -> класс
FOOD_TO_YOLO = {v: k for k, v in YOLO_TO_FOOD.items()}