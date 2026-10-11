import asyncio
import hashlib
import json
import os
import re
import zipfile
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urljoin, urlsplit, parse_qs

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from preprocessing.unzip import unzip
REPORT_TYPES = {
    2: "annual_reports",
    3: "financial_statements",
    4: "consolidated_reports",
}
def safe_name(value):
    return re.sub(r'[^\w.\-]+', '_', str(value), flags=re.UNICODE).strip('._')[:120] or 'document'


async def report_parser(
    page,
    YEARS_TO_EXPAND,
    company,
    what_looking,
    directory,
    page_url='https://www.e-disclosure.ru/portal/files.aspx?id=219&type=2',
    root=None,
):
    """Скачать документы одной страницы Интерфакса. None/[] в годах — все годы.

    YEARS_TO_EXPAND сохранён для совместимости; здесь это фильтр, не клики.
    Пустая what_looking выбирает все документы в разделе, включая приложения.
    """
    root = Path(root).expanduser() if root else Path.home() / 'Desktop/AlgoTrading'
    for value in (company, directory):
        if not value or value in {'.', '..'} or '/' in value or '\\' in value:
            raise ValueError('company и directory должны быть простыми названиями папок')

    save_dir = root / 'data/reports' / directory / company
    save_dir.mkdir(parents=True, exist_ok=True)
    years = {str(y) for y in (YEARS_TO_EXPAND or [])}
    normalize = lambda s: ' '.join(s.lower().replace('ё', 'е').split())
    search_text = normalize(what_looking or '')
    reports = []
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    manifest = save_dir / f'manifest_{stamp}.json'

    def write_manifest():
        manifest.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'Открываю: {page_url}', flush=True)
    try:
        response = await page.goto(page_url, wait_until='domcontentloaded', timeout=60_000)
    except Exception as error:
        raise RuntimeError(f'Не удалось открыть {page_url}: {error}') from error
    print(f'Страница: {page.url}; HTTP: {response.status if response else None}', flush=True)
    html = await page.content()
    if (response and response.status in {403, 429}) or 'id="REQUEST-ID"' in html:
        raise RuntimeError('Интерфакс заблокировал браузерный запрос. Сбор остановлен.')

    # Не ждём .pdf: ссылки на файлы здесь ведут на обработчик загрузки.
    links = page.locator('a[href*="FileLoad.ashx" i]')
    try:
        await links.first.wait_for(state='attached', timeout=30_000)
    except PlaywrightTimeoutError as error:
        html = await page.content()
        soup_debug = BeautifulSoup(html, "html.parser")
        body_text = normalize(soup_debug.get_text(" ", strip=True))
        debug_dir = save_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
        html_path = debug_dir / f"page_{stamp}.html"
        html_path.write_text(html, encoding="utf-8")
        try:
            await page.screenshot(path=str(debug_dir / f"page_{stamp}.png"), full_page=True, timeout=10_000)
        except Exception:
            pass
        blocked = any(word in body_text for word in (
            "forbidden", "access denied", "if you are not a bot", "captcha", "доступ ограничен"
        )) or 'id="REQUEST-ID"' in html
        if blocked:
            raise RuntimeError(f"Страница блокировки. URL: {page.url}. Диагностика: {html_path}") from error
        expected_id = parse_qs(urlsplit(page_url).query).get("id")
        actual_id = parse_qs(urlsplit(page.url).query).get("id")
        is_company_page = (
            urlsplit(page.url).path.lower().endswith("/files.aspx")
            and actual_id == expected_id
            and "карточка компании" in body_text
        )
        empty_message = any(text in body_text for text in (
            "документы отсутствуют", "нет документов", "документы не найдены",
            "информация отсутствует", "нет данных"
        ))
        empty_table = False
        for table in soup_debug.find_all("table"):
            text = normalize(table.get_text(" ", strip=True))
            if "тип документа" in text and "файл" in text:
                data_rows = []
                for row in table.find_all("tr"):
                    cells = row.find_all("td", recursive=False)
                    if cells and re.fullmatch(r"\d+", cells[0].get_text(strip=True)):
                        data_rows.append(row)
                if not data_rows:
                    empty_table = True
        if is_company_page and (empty_message or empty_table):
            write_manifest()
            print(f"{company}: пустой раздел, пропускаю. Диагностика: {html_path}", flush=True)
            return []
        raise RuntimeError(
            f"Ссылки на документы не найдены. URL: {page.url}; "
            f"заголовок: {await page.title()}. HTML и скриншот: {debug_dir}"
        ) from error

    soup = BeautifulSoup(await page.content(), 'html.parser')
    seen = set()
    for link in soup.select('a[href]'):
        href = link['href']
        if 'fileload.ashx?' not in href.lower():
            continue
        row = link.find_parent('tr')
        if row is None:
            continue
        cells = [td.get_text(' ', strip=True) for td in row.find_all('td', recursive=False)]
        if len(cells) < 3:
            continue
        title, period = cells[1], cells[2]
        year_match = re.search(r'\b(?:19|20)\d{2}\b', period)
        year = year_match.group() if year_match else None
        if years and year not in years:
            continue
        if search_text and search_text not in normalize(title):
            continue
        url = urljoin(page.url, href)
        if url in seen:
            continue
        seen.add(url)
        query = parse_qs(urlsplit(url).query)
        file_id = next((v[0] for k, v in query.items() if k.lower() == 'fileid'), '')
        reports.append(dict(company=company, document_type=directory,
                            title=title, period=period, year=year, cells=cells,
                            url=url, href=href, file_id=file_id,
                            source_url=page.url, status='pending'))

    print(f'Найдено документов: {len(reports)}')
    write_manifest()
    for index, report in enumerate(reports):
        temporary = save_dir / f'.download_{stamp}_{index}.part'
        try:
            # Настоящая браузерная загрузка с текущими cookies.
            locator = page.locator('a[href=' + json.dumps(report['href']) + ']').first
            async with page.expect_download(timeout=60_000) as download_info:
                await locator.click(timeout=30_000)
            download = await download_info.value
            await download.save_as(str(temporary))
            failure = await download.failure()
            if failure:
                raise RuntimeError(failure)

            with temporary.open('rb') as stream:
                prefix = stream.read(4096).lstrip().lower()
            if not prefix or prefix.startswith((b'<!doctype html', b'<html')) or b'<html' in prefix:
                raise RuntimeError('Вместо документа скачан HTML/пустой ответ. Сбор остановлен.')
            original = Path(download.suggested_filename.replace('\\', '/')).name
            suffix = Path(original).suffix.lower()
            if zipfile.is_zipfile(temporary):
                suffix = '.zip'
            elif prefix.startswith(b'%pdf-'):
                suffix = '.pdf'
            elif suffix not in {'.doc', '.docx', '.rtf', '.xls', '.xlsx', '.zip', '.pdf', '.7z', '.rar'}:
                raise RuntimeError(f'Неизвестный формат файла: {original}')
            digest = hashlib.sha256()
            with temporary.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            sha256 = digest.hexdigest()
            name = f"{report['year'] or 'unknown'}_{safe_name(report['title'])}_{safe_name(report['file_id'])}_{sha256[:12]}{suffix}"
            path = save_dir / name
            if path.exists():
                temporary.unlink()
                report['status'] = 'already_exists'
            else:
                temporary.replace(path)
                report['status'] = 'downloaded'
            report.update(local_path=str(path), sha256=sha256, original_filename=original)
            print(f"{report['status']}: {path.name}")
        except Exception as error:
            report.update(status='error', error=str(error))
            # После ошибки загрузки не запускаем остальные запросы вслепую.
            raise RuntimeError(f"Ошибка загрузки {report['url']}: {error}") from error
        finally:
            temporary.unlink(missing_ok=True)
            write_manifest()
        await asyncio.sleep(2)
    print(f'Папка: {save_dir}\nЖурнал: {manifest}')
    return reports


@asynccontextmanager
async def browser_session(root=None, browser_name="yandex"):
    root = Path(root).expanduser() if root else Path.home() / "Desktop/AlgoTrading"
    browser_path = Path("/Applications/Yandex.app/Contents/MacOS/Yandex")
    if browser_name not in {"yandex", "chromium"}:
        raise ValueError("browser_name: yandex или chromium")
    if browser_name == "yandex" and not browser_path.is_file():
        raise FileNotFoundError(f"Не найден браузер: {browser_path}")
    ca_path = root / "Russian_Trusted_CA.pem"
    if ca_path.is_file():
        os.environ["NODE_EXTRA_CA_CERTS"] = str(ca_path)
    print(f"Запускаю Playwright, браузер: {browser_name}", flush=True)
    async with async_playwright() as p:
        print("Playwright готов. Подключаю браузер (до 30 секунд)…", flush=True)
        options = dict(
            user_data_dir=str(root / f".browser-profile-interfax-{browser_name}"),
            headless=False, accept_downloads=True, timeout=30_000,
            chromium_sandbox=True,
        )
        if browser_name == "yandex":
            options["executable_path"] = str(browser_path)
        try:
            context = await p.chromium.launch_persistent_context(**options)
        except Exception as error:
            raise RuntimeError(
                f"Не удалось запустить/подключить {browser_name}: {error}. "
                "Если Яндекс остаётся на about:blank, установи Chromium: "
                "python -m playwright install chromium, затем укажи browser_name='chromium'."
            ) from error
        print("Браузер подключён", flush=True)
        try:
            page = context.pages[0] if context.pages else await context.new_page()
            yield page
        finally:
            await context.close()


async def parse_companies(companies, years=None, report_types=(2,), root=None, browser_name="yandex"):
    """Один браузер, один контекст и одна вкладка на весь пакет."""
    for report_type in report_types:
        if report_type not in REPORT_TYPES:
            raise ValueError(f"Неизвестный тип: {report_type}")
    result = {}
    async with browser_session(root=root, browser_name=browser_name) as page:
        for company, company_id in companies.items():
            result[company] = {}
            for report_type in report_types:
                print(f"\nКомпания: {company}, раздел: {REPORT_TYPES[report_type]}")
                result[company][report_type] = await report_parser(
                    page=page,
                    YEARS_TO_EXPAND=years,
                    company=company,
                    what_looking="",
                    directory=REPORT_TYPES[report_type],
                    page_url=f"https://www.e-disclosure.ru/portal/files.aspx?id={company_id}&type={report_type}",
                    root=root,
                )
                await asyncio.sleep(2)
    return unzip()
