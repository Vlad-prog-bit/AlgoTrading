from pathlib import Path
from datetime import datetime
import hashlib
import json
import re
from pypdf import PdfReader
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

def clean_text(text, page_number=None):
    text = re.sub(
        r"(?:https?://|www\.)\S+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    lines = text.strip().splitlines()

    if page_number is not None:
        if lines and lines[0].strip() == str(page_number):
            lines.pop(0)

        if lines and lines[-1].strip() == str(page_number):
            lines.pop()

    return "\n".join(lines)


def read_document(file_path):
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(file_path)

        parts = [
            clean_text(page.extract_text() or "", page_number=number)
            for number, page in enumerate(reader.pages, start=1)
        ]

    elif suffix == ".docx":
        document = Document(file_path)
        parts = []

        # Абзацы и таблицы в порядке расположения в документе.
        for element in document.element.body:
            if element.tag.endswith("}p"):
                paragraph = Paragraph(element, document)
                parts.append(clean_text(paragraph.text))

            elif element.tag.endswith("}tbl"):
                table = Table(element, document)

                for row in table.rows:
                    # Сохраняем табуляцию между ячейками.
                    cells = [
                        clean_text(cell.text).replace("\n", " ")
                        for cell in row.cells
                    ]
                    parts.append("\t".join(cells))

    else:
        raise ValueError(f"Неподдерживаемый формат: {suffix}")

    return "\n\n".join(part for part in parts if part.strip())


def save_document_text(
    file_path, text, save_dir, metadata_path, company, report_type, report
):
    save_dir.mkdir(parents=True, exist_ok=True)

    # Различаем одинаковые имена из разных папок и форматов.
    file_id = hashlib.sha256(
        str(file_path).encode("utf-8")
    ).hexdigest()[:12]

    txt_path = save_dir / f"{file_path.stem}_{file_id}.txt"

    temporary_text = txt_path.with_suffix(".txt.tmp")
    temporary_text.write_text(text, encoding="utf-8")
    temporary_text.replace(txt_path)

    records = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else []
    )

    # Совместимость с прежними записями source_pdf.
    by_source = {}
    for record in records:
        source = record.get("source_file") or record.get("source_pdf")
        if not source:
            raise ValueError("В метаданных нет пути исходного документа")
        by_source[str(Path(source).resolve())] = record

    by_source[str(file_path)] = {
        "company": company,
        "report_type": report_type,
        "source_file": str(file_path),
        "source_url": report.get("url"),
        "title": report.get("title"),
        "publish_time": datetime.strptime(
            report["cells"][5 if report_type == "annual_reports" else 4],
            "%d.%m.%Y",
        ).date().isoformat(),
        "text_file": txt_path.name,
        "text_path": str(txt_path),
    }

    temporary_json = metadata_path.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(
            list(by_source.values()),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_json.replace(metadata_path)

    return txt_path


def process_reports( companies, report_types):
    root = Path().home() / 'Desktop/AlgoTrading'
    reports_dir = root / "data/reports"
    texts_dir = root / "data/texts"
    texts_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = texts_dir / "metadata.json"
    processed = set()

    for company in companies:
        for report_type in report_types:
            manifest = (
                reports_dir / report_type / company / "metadata.json"
            )

            if not manifest.is_file():
                print(f"Нет метаданных: {manifest}")
                continue

            reports = json.loads(manifest.read_text(encoding="utf-8"))
            
            for report in reports:
                files = report.get("extracted_files") or []

                # Документ мог быть скачан без архива.
                if not files and report.get("local_path"):
                    files = [{"local_path": report["local_path"]}]

                for file_info in files:
                    file_path = Path(file_info["local_path"]).resolve()
                    file_name = file_path.stem
                    if file_path.suffix.lower() not in {".pdf", ".docx"}:
                        continue

                    if file_path in processed:
                        continue

                    try:
                        if not file_path.is_file():
                            raise FileNotFoundError(file_path)

                        print(f"Читаю: {file_path.name}", flush=True)
                        text = read_document(file_path)

                        if not text.strip():
                            print(
                                f"Пустой текст: {file_path.name}. "
                                "Для сканированного PDF может потребоваться OCR."
                            )
                            continue

                        txt_path = save_document_text(
                            file_path=file_path,
                            text=text,
                            save_dir=texts_dir / report_type / company,
                            metadata_path=metadata_path,
                            company=company,
                            report_type=report_type,
                            report=report,
                        )

                        processed.add(file_path)
                        print(f"Сохранён: {txt_path}")

                    except Exception as error:
                        print(f"Ошибка в {file_path}:\n{error}")
