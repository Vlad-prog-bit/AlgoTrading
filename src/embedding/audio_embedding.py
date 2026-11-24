import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path


import mlx_embeddings
import numpy as np
from tqdm.auto import tqdm
import mlx.core as mx
import gc

    
def load_model(model_path):
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Папка модели эмбеддингов не найдена: {model_path}. "
            "Укажите путь к существующей локальной модели."
        )
    return mlx_embeddings.load(str(model_path))

def find_metadata(metadata_path):
    path = Path(metadata_path)
    if path.suffix.lower() != ".json":
        path = Path(f"{path}.json")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
    
def load_text(video_path, text_file):
    return (Path(video_path) / text_file).read_text(encoding="utf-8")

def make_chunk(text, chunk_length, overelap, model, tokenizer,
               save_path,topic, source_metadata=None,
               model_path=None):
    """Сохранить векторы в NPY, а текст и позиции чанков — в metadata.json.

    Длина и перекрытие заданы в символах. Возвращает матрицу эмбеддингов.
    """
    if chunk_length <= 0:
        raise ValueError("chunk_length должен быть положительным")
    if not 0 <= overelap < chunk_length:
        raise ValueError("overelap должен быть от 0 до chunk_length - 1")
    if not text:
        raise ValueError("text не должен быть пустым")

    source_metadata = source_metadata if source_metadata is not None else {}
    video_id = source_metadata.get("video_id") or source_metadata.get("id")
    file_id = str(video_id) if video_id else hashlib.sha256(text.encode("utf-8")).hexdigest()
    if Path(file_id).name != file_id or file_id in {".", ".."}:
        raise ValueError("Некорректный ID видео для имени файла")
    output_path = Path(save_path) / f"{file_id}.npy"
    created_at = datetime.now(timezone.utc).isoformat()
    metadata_path = output_path.parent / "metadata.json"
    records = find_metadata(metadata_path) if metadata_path.exists() else []
    if not isinstance(records, list):
        raise ValueError("metadata.json должен содержать список записей")

    embeddings = []
    chunks = []
    step = chunk_length - overelap
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_length]
        output = mlx_embeddings.generate(
            model=model, processor=tokenizer, texts=[chunk],
        )
        embeddings.append(
            np.asarray(output.text_embeds.tolist(), dtype=np.float32)
        )
        chunks.append({
            "start": start,
            "end": start + len(chunk),
            "text": chunk,
        })
        if start + chunk_length >= len(text):
            break

    result = np.concatenate(embeddings, axis=0)
    if result.ndim != 2 or result.shape[0] != len(chunks):
        raise ValueError("Ожидается один вектор на каждый чанк")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "video_id": video_id,
        "topic": topic,
        "model_path": str(model_path) if model_path is not None else None,
        "embedding_path": str(output_path.resolve()),
        "embedding_shape": list(result.shape),
        "embedding_dtype": str(result.dtype),
        "text_length": len(text),
        "chunk_length": chunk_length,
        "overlap": overelap,
        "chunk_count": len(chunks),
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
def audio_embedding(model_path,video_path,metadata_path,save_path,overelap,chunk_length):
    metadata = find_metadata(metadata_path)
    if isinstance(metadata, dict):
        metadata = [metadata]
    if not metadata:
        return
    model, tokenizer = load_model(str(model_path))
    for i in tqdm(metadata,desc='Эмбеддинг видео на mlx'):
        text = load_text(video_path,i['text_file'])
        topic = i.get('topic')
        make_chunk(text=text, chunk_length=chunk_length, overelap=overelap, model = model, tokenizer=tokenizer,
               save_path=save_path, topic=topic,
               source_metadata=i, model_path=model_path)
    mx.clear_cache()
    del model 
    del tokenizer
    gc.collect()
    

