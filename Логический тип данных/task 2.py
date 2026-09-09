def help_bool(letter):
    letter = str(letter).lower().strip()

    if letter == 'к':
        return "Коммутативность (переместительный закон): A and B == B and A, а также A or B == B or A."
    elif letter == 'а':
        return "Ассоциативность (сочетательный закон): (A and B) and C == A and (B and C), аналогично для or."
    elif letter == 'д':
        return "Дистрибутивность (распределительный закон): A and (B or C) == (A and B) or (A and C), а также A or (B and C) == (A or B) and (A or C)."
    elif letter == 'м':
        return "Правило де Моргана (закон отрицания): not (A and B) == not A or not B, а также not (A or B) == not A and not B."
    else:
        return (
            "Неверный аргумент. Используйте одну из следующих букв:\n"
            "'к' — Коммутативность\n"
            "'а' — Ассоциативность\n"
            "'д' — Дистрибутивность\n"
            "'м' — Правило де Моргана"
        )
print(help_bool(1))
print(help_bool("К"))