from score_calculator import get_compound_score

# Список тестовых молекул
test_molecules = [
    ("Парацетамол", "CC(=O)NC1=CC=C(O)C=C1"),
    ("Ибупрофен", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
    ("Аспирин", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("Метформин", "CN(C)C(=N)NC(=N)N"),
    ("Атропин", "CN1C2CCC1CC(C2)OC(=O)C(CO)C3=CC=CC=C3"),
    ("Кофеин", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
]

print("Тестирование функции расчета скора для различных молекул\n")
print("%-20s %-10s" % ("Название", "Скор"))
print("-" * 32)

for name, smiles in test_molecules:
    score = get_compound_score(smiles)
    if score is not None:
        print("%-20s %-10.4f" % (name, score))
    else:
        print("%-20s %-10s" % (name, "Ошибка")) 