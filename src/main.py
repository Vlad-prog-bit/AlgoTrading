"""AlgoTrading: каркас исследовательского проекта."""
import asyncio
import platform
from pathlib import Path
from tqdm.auto import tqdm
from embedding.audio_embedding import audio_embedding
from embedding.text_embedding import text_embedding
from parser.report_parser import parse_companies
from parser.video_parser import video_parser
from preprocessing.text_reader import process_reports
from parser.parser_of_pulse import pulse_parser

system = platform.system()
'''
COMPANIES = {
    # Банки
    "sber": 3043,
    "tbank": 2989,       # АО «ТБанк», не холдинг

    # Нефть и газ
    "gazprom": 934,
    "lukoil": 17,
    "novatek": 225,

    # Металлургия и добыча
    "severstal": 30,
    "nornickel": 564,
    "polyus": 7832,

    # Розничная торговля
    "magnit": 7671,

    # Технологии и телеком
    "vk": 38965,         # МКПАО «ВК»
    "megafon": 219,
}
'''
#COMPANIES = {"sber": 3043,
#             "tbank": 2989}
COMPANIES = {"sber": 3043}
'''
VIDEO_TOPICS = [
    # Отчётность
    "анализ финансовой отчетности компаний МСФО",
    "связь баланса отчета о прибылях и движении денежных средств",
    "анализ денежного потока FCF операционный денежный поток",
    "качество прибыли разовые доходы финансовая отчетность",
    "долговая нагрузка компании чистый долг EBITDA",

    # Оценка компаний
    "оценка акций мультипликаторы PE PB EV EBITDA",
    "DCF оценка компании дисконтирование денежных потоков",
    "анализ рентабельности ROE ROIC",
    "дивиденды свободный денежный поток устойчивость выплат",

    # Особенности отраслей
    "анализ отчетности банков процентная маржа стоимость риска",
    "анализ нефтегазовых компаний цены на нефть курс рубля",
    "анализ металлургических компаний цикличность",
    "анализ ритейла сопоставимые продажи LFL",

    # Проверяемые торговые гипотезы
    "фактор momentum исследование доходности акций",
    "факторы value quality size акции",
    "реакция акций на публикацию финансовой отчетности",
    "возврат к среднему торговые стратегии исследование",

    # ML и проверка стратегий
    "машинное обучение финансовые временные ряды признаки",
    "CatBoost прогнозирование доходности акций",
    "walk forward validation финансовые временные ряды",
    "утечка данных look ahead bias survivorship bias бэктестинг",
    "бэктестинг стратегии комиссии проскальзывание",
    "переобучение торговых стратегий out of sample",

    # Риск
    "управление риском размер позиции волатильность",
    "оценка стратегии Sharpe Sortino максимальная просадка",
]
'''
VIDEO_TOPICS = ["анализ финансовой отчетности компаний МСФО"]
root = Path(__file__).resolve().parents[1]
async def start_parser():
    '''
    здесь начинается вызов
    функций парсинга
    '''
    #report_types=(2,3,4)
    report_types=(4,)
    years=list(range(2025,2026))
    companies_to_parse = COMPANIES.keys()
    reports_path = root / "data/reports"
    return await parse_companies(
        companies={name: COMPANIES[name] for name in companies_to_parse},
        years=years,  # None — все годы
        report_types=report_types,  # 2 — годовые, 3 — бухгалтерские, 4 — консолидированные
        reports_path = reports_path
    )
if __name__ == '__main__':
    asyncio.run(start_parser())
    for topic in tqdm(VIDEO_TOPICS):
        save_path = root / "data/audio"
        video_parser(count=2,
                     topic=topic,
                     save_path=save_path,
                    )
    pulse_parser(ROOT=root,
                 DATA_PATH=root / "data/tpulse",
                 SITEMAPS_PATH=root / "data/tpulse/sitemaps",
                 PARQUET_PATH=root / "data/tpulse/posts.parquet",
                 CA_PATH=root / "Russian_Trusted_CA.pem",
                 CA_BUNDLE=root / "data/tpulse/ca_bundle.pem",
                 INDEX_URL="https://www.tbank.ru/sitemap/invest/pulse.xml.gz",
                 limit=1, # Например, 10 для проверки; None — все
                 REQUEST_DELAY=1,
                 BATCH_SIZE=100)

    if system == "Darwin":
        from preprocessing.transcript_mac import transcript_mac
    
        transcript_mac(
            root = root,
            audio_path = root / 'data/audio',
            model_path = root / "Models/whisper-mlx",
            save_path = root / "data/transcripts",
            metadata_path = root / "data/audio/metadata.json"
        )
    else:
        from preprocessing.transcript_windows import transcript_windows
        transcript_windows()
    process_reports(
        companies=COMPANIES.keys(),
        report_types=[
            "annual_reports",
            "financial_statements",
            "consolidated_reports",
            ],
            reports_path = root / "data/reports",
            texts_path = root / "data/texts",
            metadata_path = root / "data/texts/metadata.json",
            root=root
            )
            
    audio_embedding(model_path=root / "Models/Qwen3-Embedding-8B-4bit",
                    video_path = root / "data/transcripts",
                    metadata_path = root / "data/transcripts/metadata.json",
                    save_path = root / "data/vectors/transcripts",
                    chunk_length = 1400,
                    overelap = 50)
    
    text_embedding(texts_path = root / "data/texts",
                   save_path = root / "data/vectors/texts", 
                   model_path=root / "Models/Qwen3-Embedding-8B-4bit",
                   chunk_length = 1400, 
                   metadata_path = root / "data/texts/metadata.json",
                   overlap = 50)





    
    print('Конец выполнения программы')