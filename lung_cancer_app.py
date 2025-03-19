import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib

# Загрузка модели и скейлера
model = joblib.load("lung_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")


# Функция для предсказания
def predict():
    try:
        age = age_entry.get()
        if not age.isdigit():
            raise ValueError("Возраст должен быть числом.")

        data = {
            'GENDER': 2 if gender_var.get() == 'Мужской' else 1,
            'AGE': int(age),
            'SMOKING': 2 if smoking_var.get() == 'Да' else 1,
            'YELLOW_FINGERS': 2 if yellow_fingers_var.get() == 'Да' else 1,
            'ANXIETY': 2 if anxiety_var.get() == 'Да' else 1,
            'PEER_PRESSURE': 2 if peer_pressure_var.get() == 'Да' else 1,
            'CHRONIC DISEASE': 2 if chronic_disease_var.get() == 'Да' else 1,
            'FATIGUE ': 2 if fatigue_var.get() == 'Да' else 1,
            'ALLERGY ': 2 if allergy_var.get() == 'Да' else 1,
            'WHEEZING': 2 if wheezing_var.get() == 'Да' else 1,
            'ALCOHOL CONSUMING': 2 if alcohol_var.get() == 'Да' else 1,
            'COUGHING': 2 if coughing_var.get() == 'Да' else 1,
            'SHORTNESS OF BREATH': 2 if breath_var.get() == 'Да' else 1,
            'SWALLOWING DIFFICULTY': 2 if swallowing_var.get() == 'Да' else 1,
            'CHEST PAIN': 2 if chest_pain_var.get() == 'Да' else 1
        }

        # Отладка: вывод данных
        print("Вводимые данные:", data)

        # Создание DataFrame
        df = pd.DataFrame([data])
        print("\nDataFrame перед масштабированием:\n", df)

        # Масштабирование данных
        df_scaled = scaler.transform(df)
        print("\nDataFrame после масштабирования:\n", df_scaled)

        # Предсказание
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0][1]

        # Вывод результата
        result = f"Вероятность рака лёгких: {probability * 100:.2f}%"
        if prediction == 1:
            messagebox.showinfo("Результат", f"Риск: Высокий.\n{result}")
        else:
            messagebox.showinfo("Результат", f"Риск: Низкий.\n{result}")

    except ValueError as ve:
        messagebox.showerror("Ошибка ввода", f"Проверьте правильность данных.\n\n{ve}")
    except Exception as e:
        messagebox.showerror("Ошибка", f"Что-то пошло не так.\n\n{str(e)}")


# Создание главного окна
root = tk.Tk()
root.title("Прогноз рака лёгких")
root.geometry("400x700")

# Полоса прокрутки
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)

canvas = tk.Canvas(main_frame)
scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)


# Функция для полей ввода
def create_input(label, variable):
    frame = ttk.Frame(scrollable_frame)
    frame.pack(fill=tk.X, pady=2)
    ttk.Label(frame, text=label, width=25).pack(side=tk.LEFT)
    ttk.Combobox(frame, textvariable=variable, values=['Да', 'Нет']).pack(side=tk.RIGHT)


# Поля ввода
gender_var = tk.StringVar(value='Мужской')
smoking_var = tk.StringVar(value='Нет')
yellow_fingers_var = tk.StringVar(value='Нет')
anxiety_var = tk.StringVar(value='Нет')
peer_pressure_var = tk.StringVar(value='Нет')
chronic_disease_var = tk.StringVar(value='Нет')
fatigue_var = tk.StringVar(value='Нет')
allergy_var = tk.StringVar(value='Нет')
wheezing_var = tk.StringVar(value='Нет')
alcohol_var = tk.StringVar(value='Нет')
coughing_var = tk.StringVar(value='Нет')
breath_var = tk.StringVar(value='Нет')
swallowing_var = tk.StringVar(value='Нет')
chest_pain_var = tk.StringVar(value='Нет')

# Интерфейс
ttk.Label(scrollable_frame, text="Прогноз рака лёгких", font=("Helvetica", 16)).pack(pady=10)

# Поле возраста
frame = ttk.Frame(scrollable_frame)
frame.pack(fill=tk.X, pady=2)
ttk.Label(frame, text="Возраст", width=25).pack(side=tk.LEFT)
age_entry = ttk.Entry(frame)
age_entry.pack(side=tk.RIGHT)

# Поле пола
frame = ttk.Frame(scrollable_frame)
frame.pack(fill=tk.X, pady=2)
ttk.Label(frame, text="Пол", width=25).pack(side=tk.LEFT)
ttk.Combobox(frame, textvariable=gender_var, values=['Мужской', 'Женский']).pack(side=tk.RIGHT)

# Другие поля
create_input("Курение", smoking_var)
create_input("Жёлтые пальцы", yellow_fingers_var)
create_input("Тревожность", anxiety_var)
create_input("Давление сверстников", peer_pressure_var)
create_input("Хронические заболевания", chronic_disease_var)
create_input("Усталость", fatigue_var)
create_input("Аллергия", allergy_var)
create_input("Свистящее дыхание", wheezing_var)
create_input("Употребление алкоголя", alcohol_var)
create_input("Кашель", coughing_var)
create_input("Одышка", breath_var)
create_input("Трудности с глотанием", swallowing_var)
create_input("Боль в груди", chest_pain_var)

# Кнопка предсказания
ttk.Button(scrollable_frame, text="Провести анализ", command=predict).pack(pady=10)

# Настройка скролла
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

root.mainloop()
