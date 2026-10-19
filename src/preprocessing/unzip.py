import json
from pathlib import Path
from zipfile import ZipFile


def unzip():
    reports_dir = Path.home() / "Desktop/AlgoTrading/data/reports"

    for archive_path in sorted(reports_dir.rglob("*.zip")):
        try:
            destination = archive_path.parent.resolve()
            metadata_path = destination / "metadata.json"

            records = (
                json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.exists()
                else []
            )

            matching_records = [
                record
                for record in records
                if record.get("local_path")
                and Path(record["local_path"]).resolve()
                == archive_path.resolve()
            ]

            if not matching_records:
                raise ValueError("В metadata.json нет записи этого архива")

            extracted_files = []

            with ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    target = (destination / member.filename).resolve()

                    if not target.is_relative_to(destination):
                        raise ValueError(
                            f"Недопустимый путь: {member.filename}"
                        )

                    if member.is_dir():
                        continue

                    if target.exists():
                        raise FileExistsError(
                            f"Файл уже существует: {target}"
                        )

                    extracted_files.append({
                        "name": target.name,
                        "relative_path": member.filename,
                        "local_path": str(target),
                    })

                damaged_file = archive.testzip()
                if damaged_file is not None:
                    raise ValueError(
                        f"Повреждённый файл: {damaged_file}"
                    )

                archive.extractall(destination)

            for record in matching_records:
                record["extracted_files"] = extracted_files

            temporary = metadata_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(metadata_path)

            # Удаляем ZIP только после сохранения метаданных.
            archive_path.unlink()
            print(f"Распакован: {archive_path.name}")

        except Exception as error:
            print(f"Архив сохранён: {archive_path}\nПричина: {error}")