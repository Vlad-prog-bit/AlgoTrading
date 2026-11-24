import gzip
import re
import shutil
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID

import certifi
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from bs4 import BeautifulSoup




def prepare_certificates(DATA_PATH,
                         CA_BUNDLE,
                         CA_PATH):
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    CA_BUNDLE.write_bytes(
        Path(certifi.where()).read_bytes()
        + b"\n"
        + CA_PATH.read_bytes()
        + b"\n"
    )


def download_xml(session, 
                url,
                save_path,
                CA_BUNDLE):
    """Скачать gzip и сохранить распакованный XML."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = save_path.with_suffix(".xml.tmp")

    try:
        with session.get(
            url,
            verify=str(CA_BUNDLE),
            stream=True,
            timeout=(15, 60),
        ) as response:
            response.raise_for_status()

            with gzip.GzipFile(fileobj=response.raw) as archive:
                with temporary.open("wb") as file:
                    shutil.copyfileobj(archive, file)

        # Проверяем, что получен корректный XML.
        for _, element in ET.iterparse(temporary, events=("end",)):
            element.clear()

        temporary.replace(save_path)

    finally:
        temporary.unlink(missing_ok=True)


def read_sitemap_urls(file_path):
    """Последовательно извлекать ссылки из XML."""
    with Path(file_path).open("rb") as file:
        for _, element in ET.iterparse(file, events=("end",)):
            if element.tag.rsplit("}", 1)[-1] == "loc":
                url = (element.text or "").strip()
                if url:
                    yield url

            element.clear()


def get_post_id(url):
    parsed = urlparse(url)

    if (
        parsed.scheme != "https"
        or parsed.hostname != "www.tbank.ru"
        or not parsed.path.startswith("/invest/social/profile/")
    ):
        return None

    candidate = parsed.path.rstrip("/").split("/")[-1]

    try:
        return str(UUID(candidate))
    except ValueError:
        return None


def parse_post(session, 
               url, 
               post_id,
               CA_BUNDLE,
               DATE_PATTERN,
               MONTHS):
    """Извлечь текст и дату публикации из карточки поста."""
    with session.get(
        url,
        verify=str(CA_BUNDLE),
        timeout=(15, 60),
    ) as response:
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

    post = soup.find(attrs={
        "data-qa-tag": "PulsePost",
        "data-post-id": post_id,
    })

    if post is None:
        raise ValueError("Пост не найден в HTML ответа")

    body = post.find(attrs={"data-qa-tag": "PulsePostBody"})
    author = post.find(attrs={"data-qa-tag": "PulsePostAuthor"})

    if body is None or author is None:
        raise ValueError("Не найден текст или блок автора")

    for br in body.find_all("br"):
        br.replace_with("\n")

    text = body.get_text().strip()

    if not text:
        raise ValueError("Пустой текст поста")

    author_text = author.get_text(" ", strip=True)
    match = DATE_PATTERN.search(author_text)

    if match is None:
        raise ValueError(f"Не распознана дата: {author_text}")

    day, month, year, hour, minute = match.groups()

    published_at = datetime(
        year=int(year),
        month=MONTHS[month.lower()],
        day=int(day),
        hour=int(hour),
        minute=int(minute),
    )

    return {
        "post_id": post_id,
        "url": url,
        "author": urlparse(url).path.rstrip("/").split("/")[-2],
        "published_at": published_at.isoformat(),
        "published_at_raw": match.group(0),
        # Часовой пояс страницы пока не установлен.
        "timezone": None,
        "text": text,
    }


def download_sitemaps(session,
                      SITEMAPS_PATH,
                      DATA_PATH,
                      INDEX_URL,
                      REQUEST_DELAY,
                      CA_BUNDLE):
    index_path = DATA_PATH / "pulse.xml"
    download_xml(session = session,
                url=INDEX_URL,
                save_path=index_path,
                CA_BUNDLE = CA_BUNDLE)

    archive_urls = list(dict.fromkeys(
        url
        for url in read_sitemap_urls(index_path)
        if urlparse(url).path.endswith(".gz")
    ))

    (DATA_PATH / "urls.txt").write_text(
        "\n".join(archive_urls),
        encoding="utf-8",
    )

    print(f"Найдено архивов: {len(archive_urls)}")
    sitemap_files = []

    for index, url in enumerate(archive_urls, start=1):
        filename = Path(urlparse(url).path).name.removesuffix(".gz")
        save_path = SITEMAPS_PATH / filename

        if save_path.exists():
            sitemap_files.append(save_path)
            continue

        try:
            download_xml(session = session, 
                         url = url, 
                         save_path = save_path,
                         CA_BUNDLE=CA_BUNDLE)
            sitemap_files.append(save_path)
            print(f"[{index}/{len(archive_urls)}] Скачан: {filename}")

        except (
            requests.RequestException,
            OSError,
            EOFError,
            ET.ParseError,
        ) as error:
            print(f"Ошибка sitemap: {url}\n{error}")

        time.sleep(REQUEST_DELAY)

    return sitemap_files


def download_posts(session,
                   sitemap_files, 
                   PARQUET_PATH, 
                   limit,
                   POST_SCHEMA,
                   REQUEST_DELAY,
                   BATCH_SIZE,
                   CA_BUNDLE,
                   DATE_PATTERN,
                   MONTHS):
    temporary = PARQUET_PATH.with_suffix(".parquet.tmp")
    seen = set()
    buffer = []
    attempted = 0
    saved = 0

    try:
        with pq.ParquetWriter(
            temporary,
            schema=POST_SCHEMA,
            compression="zstd",
        ) as writer:

            # Переносим существующие записи в новый файл частями.
            if PARQUET_PATH.exists():
                with pq.ParquetFile(PARQUET_PATH) as existing:
                    for batch in existing.iter_batches(batch_size=1000):
                        table = pa.Table.from_batches([batch]).cast(
                            POST_SCHEMA
                        )
                        seen.update(table["post_id"].to_pylist())
                        writer.write_table(table)

                print(f"Уже сохранено постов: {len(seen)}")

            stop = False

            for sitemap_path in sitemap_files:
                if stop:
                    break

                for url in read_sitemap_urls(sitemap_path):
                    post_id = get_post_id(url)

                    if post_id is None or post_id in seen:
                        continue

                    if limit is not None and attempted >= limit:
                        stop = True
                        break

                    seen.add(post_id)
                    attempted += 1

                    try:
                        post = parse_post(session, url, post_id, CA_BUNDLE, DATE_PATTERN, MONTHS)

                    except (
                        requests.RequestException,
                        ValueError,
                    ) as error:
                        print(f"[{attempted}] Ошибка: {url}\n{error}")

                    else:
                        buffer.append(post)
                        saved += 1
                        print(
                            f"[{attempted}] Получен: {post_id} "
                            f"— {post['published_at']}"
                        )

                    if len(buffer) >= BATCH_SIZE:
                        writer.write_table(
                            pa.Table.from_pylist(
                                buffer,
                                schema=POST_SCHEMA,
                            )
                        )
                        buffer.clear()

                    time.sleep(REQUEST_DELAY)

            if buffer:
                writer.write_table(
                    pa.Table.from_pylist(buffer, schema=POST_SCHEMA)
                )

        # Основной файл заменяется только после успешного завершения.
        temporary.replace(PARQUET_PATH)

    finally:
        temporary.unlink(missing_ok=True)

    print(f"\nПопыток: {attempted}, добавлено: {saved}")
    print(f"Сохранено: {PARQUET_PATH.resolve()}")


def main(SITEMAPS_PATH,
        limit,
        PARQUET_PATH,
        POST_SCHEMA,
        REQUEST_DELAY,
        BATCH_SIZE,
        DATA_PATH,
        CA_BUNDLE,
        CA_PATH,
        INDEX_URL,
        DATE_PATTERN,
        MONTHS,
        ):
    prepare_certificates(DATA_PATH=DATA_PATH,
                         CA_BUNDLE=CA_BUNDLE,
                         CA_PATH=CA_PATH)
    SITEMAPS_PATH.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        sitemap_files = download_sitemaps(session=session,
                                          DATA_PATH=DATA_PATH,
                                          INDEX_URL=INDEX_URL,
                                          REQUEST_DELAY=REQUEST_DELAY,
                                          SITEMAPS_PATH=SITEMAPS_PATH,
                                          CA_BUNDLE=CA_BUNDLE)
        download_posts(session = session,
                    sitemap_files = sitemap_files,
                    PARQUET_PATH=PARQUET_PATH,
                    limit=limit,
                    POST_SCHEMA=POST_SCHEMA,
                    REQUEST_DELAY=REQUEST_DELAY,
                    BATCH_SIZE=BATCH_SIZE,
                    CA_BUNDLE=CA_BUNDLE,
                    DATE_PATTERN=DATE_PATTERN,
                    MONTHS=MONTHS)
'''
def pulse_parser(ROOT,
                 POST_LIMIT,
                 REQUEST_DELAY,
                 BATCH_SIZE):
'''
def pulse_parser(ROOT, 
    DATA_PATH,
    SITEMAPS_PATH,
    PARQUET_PATH,
    CA_PATH,
    CA_BUNDLE,
    INDEX_URL,
    limit,  
    REQUEST_DELAY,
    BATCH_SIZE):    
    MONTHS = {
        "января": 1,
        "февраля": 2,
        "марта": 3,
        "апреля": 4,
        "мая": 5,
        "июня": 6,
        "июля": 7,
        "августа": 8,
        "сентября": 9,
        "октября": 10,
        "ноября": 11,
        "декабря": 12,
    }

    DATE_PATTERN = re.compile(
        r"(\d{1,2})\s+"
        r"(января|февраля|марта|апреля|мая|июня|июля|"
        r"августа|сентября|октября|ноября|декабря)"
        r"\s+(\d{4})\s+в\s+(\d{1,2}):(\d{2})",
        flags=re.IGNORECASE,
    )

    POST_SCHEMA = pa.schema([
        ("post_id", pa.string()),
        ("url", pa.string()),
        ("author", pa.string()),
        ("published_at", pa.string()),
        ("published_at_raw", pa.string()),
        ("timezone", pa.string()),
        ("text", pa.string()),
    ])
    main(SITEMAPS_PATH = SITEMAPS_PATH ,
        limit = limit,
        PARQUET_PATH = PARQUET_PATH,
        POST_SCHEMA = POST_SCHEMA,
        REQUEST_DELAY = REQUEST_DELAY,
        BATCH_SIZE = BATCH_SIZE,
        DATA_PATH = DATA_PATH,
        CA_BUNDLE = CA_BUNDLE,
        CA_PATH = CA_PATH,
        INDEX_URL = INDEX_URL,
        DATE_PATTERN=DATE_PATTERN,
        MONTHS=MONTHS)


