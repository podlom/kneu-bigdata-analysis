import pypandoc
import sys
import os


def main():
    # Перевірка, чи було передано аргументи командного рядка
    if len(sys.argv) < 2:
        # Отримання імені поточного скрипта
        script_file_name = os.path.basename(sys.argv[0])

        print("Використання: python3 ", script_file_name, " filename.md")
        sys.exit(1)  # Вихід із скрипта з помилкою

    # Отримання назви файлу з аргументів командного рядка
    filename = sys.argv[1]

    # Формування імені вихідного файлу, замінюючи розширення .md на .docx
    base_filename, _ = os.path.splitext(filename)
    output_filename = base_filename + ".docx"

    # Тут ваш код для обробки MD файлу
    try:
        output = pypandoc.convert_file(filename, 'docx', outputfile=output_filename)
        assert output == ""
    except Exception as e:
        print(f"Помилка при читанні файлу {filename}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

