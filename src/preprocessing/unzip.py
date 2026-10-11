from pathlib import Path
from zipfile import ZipFile
def unzip():
    '''
    в это фрагменте кода происходит разархивирование файлов в reports 
    с последующим удлением архива
    '''
    reports_dir = Path("/Users/vladislav/Desktop/AlgoTrading/data/reports")
    archives = sorted(
        path for path in reports_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".zip"
    )
    for archive_path in archives:
        try:
            with ZipFile(archive_path) as archive:
                destination = archive_path.parent.resolve()
                # Проверяем пути и существующие файлы до распаковки.
                for member in archive.infolist():
                    target = (destination / member.filename).resolve()

                    if not target.is_relative_to(destination):
                        raise ValueError(f"Недопустимый путь: {member.filename}")

                    if not member.is_dir() and target.exists():
                        raise FileExistsError(f"Файл уже существует: {target}")

                damaged_file = archive.testzip()
                if damaged_file is not None:
                    raise ValueError(f"Повреждённый файл: {damaged_file}")

                archive.extractall(destination)

            archive_path.unlink()
            print(f"Распакован, архив удалён: {archive_path}")

        except Exception as error:
            print(f"Архив сохранён: {archive_path}\n  Причина: {error}")