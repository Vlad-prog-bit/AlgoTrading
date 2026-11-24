import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx_embeddings
import numpy as np
from tqdm.auto import tqdm


def load_model(model_path):
    return mlx_embeddings.load(model_path)

def find_metadata(metadata_path):
    path = Path(metadata_path)
    if path.suffix.lower() != ".json":
        path = Path(f"{path}.json")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
    
def load_text(texts_path, text_file):
    return (Path(texts_path) / text_file).read_text(encoding="utf-8")

def make_chunk(text, chunk_length, overelap, model, tokenizer,
               save_path, source_metadata=None,
               model_path=None):
    """Сохранить матрицу эмбеддингов; длина и перекрытие заданы в символах."""
    if chunk_length <= 0:
        raise ValueError("chunk_length должен быть положительным")
    if not 0 <= overelap < chunk_length:
        raise ValueError("overelap должен быть от 0 до chunk_length - 1")
    if not text.strip():
        raise ValueError("text не должен быть пустым")

    source_metadata = source_metadata if source_metadata is not None else {}
    source_id = source_metadata.get("source_file") or source_metadata.get("text_path")
    file_id = (
        hashlib.sha256(str(source_id).encode("utf-8")).hexdigest()
        if source_id else hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    if Path(file_id).name != file_id or file_id in {".", ".."}:
        raise ValueError("Некорректный ID отчёта для имени файла")
    output_path = Path(save_path) / f"{file_id}.npy"
    metadata_path = output_path.parent / "metadata.json"
    records = find_metadata(metadata_path) if metadata_path.exists() else []
    if not isinstance(records, list):
        raise ValueError("metadata.json должен содержать список записей")

    embeddings = []
    chunks = []
    step = chunk_length - overelap
    chunk_count = 1 + (max(0, len(text) - chunk_length) + step - 1) // step
    for index in tqdm(
        range(chunk_count), desc="Фрагменты", unit="фрагм.", leave=False,
    ):
        start = index * step
        chunk = text[start:start + chunk_length]
        output = mlx_embeddings.generate(
            model=model, processor=tokenizer, texts=[chunk],
        )
        embeddings.append(
            np.asarray(output.text_embeds.tolist(), dtype=np.float32)
        )
        chunks.append({
            "index": len(chunks),
            "start": start,
            "end": start + len(chunk),
            "text": chunk,
        })

    result = np.concatenate(embeddings, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        **source_metadata,
        "document_id": file_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path) if model_path is not None else None,
        "embedding_file": output_path.name,
        "embedding_path": str(output_path.resolve()),
        "embedding_shape": list(result.shape),
        "embedding_dtype": str(result.dtype),
        "text_length": len(text),
        "chunk_length": chunk_length,
        "overlap": overelap,
        "chunks": chunks,

    }
    records = [
        existing for existing in records
        if existing.get("embedding_file") != output_path.name
    ]
    records.append(record)
    # Проверяем сериализацию до записи эмбеддингов.
    metadata_json = json.dumps(records, ensure_ascii=False, indent=2)
    np.save(output_path, result)
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(metadata_json, encoding="utf-8")
    temporary.replace(metadata_path)
    return result
def text_embedding(metadata_path, 
                   texts_path, 
                   save_path, 
                   model_path=None,
                   chunk_length=1500, 
                   overlap=400):
    """Преобразовать тексты отчётов из metadata.json в матрицы эмбеддингов."""
    metadata = find_metadata(metadata_path)
    if isinstance(metadata, dict):
        metadata = [metadata]
    if not isinstance(metadata, list):
        raise ValueError("metadata.json должен содержать список записей")
    if not metadata:
        return

    print(f"Загрузка модели: {model_path}", flush=True)
    model, tokenizer = load_model(str(model_path))
    print("Модель загружена", flush=True)
    try:
        for record in tqdm(metadata, desc="Эмбеддинг текстов на MLX"):
            text_path = record.get("text_path")
            if text_path:
                text_path = Path(text_path)
                if not text_path.is_absolute():
                    text_path = texts_path / text_path
            else:
                text_path = texts_path
                if record.get("report_type") and record.get("company"):
                    text_path = text_path / record["report_type"] / record["company"]
                text_path = text_path / record["text_file"]
            text = text_path.read_text(encoding="utf-8")
            if not text.strip():
                tqdm.write(f"Пропущен пустой текст: {text_path}")
                continue
            step = chunk_length - overlap
            chunk_count = 1 + (max(0, len(text) - chunk_length) + step - 1) // step
            tqdm.write(
                f"Обработка: {text_path.name} — {len(text):,} символов, "
                f"{chunk_count:,} фрагментов"
            )
            make_chunk(
                text=text, chunk_length=chunk_length, overelap=overlap,
                model=model, tokenizer=tokenizer, save_path=save_path,
                source_metadata={**record, "text_path": str(text_path.resolve())},
                model_path=model_path,
            )
    finally:
        del model, tokenizer
        gc.collect()
        mx.clear_cache()



